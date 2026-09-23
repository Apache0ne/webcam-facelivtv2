// Windows GPU video surfaces -> CUDA interop -> Direct3D preview.
// This path never locks a camera buffer or downloads a full video frame.
#include <windows.h>
#include <wrl/client.h>
#include <d3d11.h>
#include <d3d10.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <d2d1.h>
#include <dwrite.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>

using Microsoft::WRL::ComPtr;
#define VIDEO_API extern "C" __declspec(dllexport)

namespace {
void check(HRESULT code, const char* operation) {
    if (FAILED(code)) {
        std::ostringstream message;
        message << operation << " failed (HRESULT 0x" << std::hex << static_cast<unsigned long>(code) << ")";
        throw std::runtime_error(message.str());
    }
}
void cuda_check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
void error_text(char* out, size_t capacity, const char* text) {
    if (!out || !capacity) return;
    size_t n = std::min(capacity - 1, std::strlen(text));
    std::memcpy(out, text, n); out[n] = 0;
}
struct Platform {
    bool com = false, mf = false;
    Platform() {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (hr != RPC_E_CHANGED_MODE) { check(hr, "CoInitializeEx"); com = true; }
        hr = MFStartup(MF_VERSION);
        if (FAILED(hr)) { if(com) CoUninitialize(); check(hr, "MFStartup"); }
        mf = true;
    }
    ~Platform() { if(mf) MFShutdown(); if(com) CoUninitialize(); }
};
struct Overlay {
    float box[4], landmarks[10], red, green, blue;
    char label[128];
};
struct Video {
    Platform platform; // Destroy COM/MF after all the interfaces below.
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IMFDXGIDeviceManager> manager;
    ComPtr<IMFMediaSource> source;
    ComPtr<IMFSourceReader> reader;
    ComPtr<IDXGISwapChain> swap;
    ComPtr<ID3D11RenderTargetView> target;
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> texture_view;
    ComPtr<ID3D11VertexShader> vertex_shader;
    ComPtr<ID3D11PixelShader> pixel_shader;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID2D1Factory> d2d;
    ComPtr<ID2D1RenderTarget> overlay_target;
    ComPtr<ID2D1SolidColorBrush> brush;
    ComPtr<IDWriteFactory> write;
    ComPtr<IDWriteTextFormat> font;
    cudaGraphicsResource* resource = nullptr;
    unsigned width = 640, height = 480, view_width = 640, view_height = 480;
    DXGI_FORMAT texture_format = DXGI_FORMAT_UNKNOWN;
    bool have_frame = false;

    ~Video() {
        if(resource) cudaGraphicsUnregisterResource(resource);
        reader.Reset();
        if(source) source->Shutdown();
        if(context) { context->ClearState(); context->Flush(); }
    }

    void open(int index, HWND window, bool camera = true) {
        if(!IsWindow(window)) throw std::runtime_error("GPU preview requires a valid native window handle");
        cuda_check(cudaSetDevice(0), "select CUDA GPU");
        ComPtr<IDXGIFactory1> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> selected;
        for(UINT i=0;;++i) {
            ComPtr<IDXGIAdapter1> adapter;
            if(factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            int ordinal = -1;
            if(cudaD3D11GetDevice(&ordinal, adapter.Get()) == cudaSuccess && ordinal == 0) {
                selected = adapter; break;
            }
        }
        if(!selected) throw std::runtime_error("No Direct3D adapter matches CUDA GPU 0");
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
        D3D_FEATURE_LEVEL feature;
        check(D3D11CreateDevice(selected.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
                               nullptr, 0, D3D11_SDK_VERSION, &device, &feature, &context), "D3D11CreateDevice");
        ComPtr<ID3D10Multithread> multithread;
        check(context.As(&multithread), "D3D multithread interface");
        multithread->SetMultithreadProtected(TRUE);
        UINT token;
        check(MFCreateDXGIDeviceManager(&token, &manager), "MFCreateDXGIDeviceManager");
        check(manager->ResetDevice(device.Get(), token), "MF GPU device manager");
        if(camera) open_camera(index);
        open_display(factory.Get(), window);
    }

    void open_camera(int index) {
        ComPtr<IMFAttributes> attributes;
        check(MFCreateAttributes(&attributes, 2), "camera attributes");
        check(attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID), "camera type");
        IMFActivate** devices = nullptr;
        UINT count = 0;
        check(MFEnumDeviceSources(attributes.Get(), &devices, &count), "enumerate cameras");
        ComPtr<IMFActivate> selected;
        if(index >= 0 && static_cast<UINT>(index) < count) selected = devices[index];
        for(UINT i=0;i<count;++i) devices[i]->Release();
        CoTaskMemFree(devices);
        if(!selected) throw std::runtime_error("Camera index is not available through Media Foundation");
        check(selected->ActivateObject(IID_PPV_ARGS(&source)), "open camera");
        check(MFCreateAttributes(&attributes, 5), "reader attributes");
        check(attributes->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, manager.Get()), "reader GPU manager");
        check(attributes->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, TRUE), "GPU video processing");
        check(attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE), "hardware video transforms");
        check(attributes->SetUINT32(MF_READWRITE_D3D_OPTIONAL, FALSE), "require GPU surfaces");
        check(MFCreateSourceReaderFromMediaSource(source.Get(), attributes.Get(), &reader), "create GPU camera reader");
        check(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE), "deselect streams");
        check(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE), "select video");
        ComPtr<IMFMediaType> type;
        check(MFCreateMediaType(&type), "video type");
        check(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "video major type");
        check(type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32), "BGRA video subtype");
        check(MFSetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, width, height), "video dimensions");
        check(MFSetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, 30, 1), "video rate");
        check(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, type.Get()), "set 640x480 GPU video output");
    }

    void open_display(IDXGIFactory1* factory, HWND window) {
        RECT rect{}; GetClientRect(window, &rect);
        view_width = std::max(1L, rect.right); view_height = std::max(1L, rect.bottom);
        DXGI_SWAP_CHAIN_DESC desc{};
        desc.BufferDesc.Width = view_width; desc.BufferDesc.Height = view_height;
        desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1; desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = 2; desc.OutputWindow = window; desc.Windowed = TRUE;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        check(factory->CreateSwapChain(device.Get(), &desc, &swap), "create GPU preview swap chain");
        factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
        ComPtr<ID3D11Texture2D> backbuffer;
        check(swap->GetBuffer(0, IID_PPV_ARGS(&backbuffer)), "preview back buffer");
        check(device->CreateRenderTargetView(backbuffer.Get(), nullptr, &target), "preview render target");
        const char* shader =
            "struct V{float4 p:SV_POSITION;float2 uv:TEXCOORD0;};"
            "V vs(uint id:SV_VertexID){V o;float2 uv=float2((id<<1)&2,id&2);o.p=float4(uv*float2(2,-2)+float2(-1,1),0,1);o.uv=uv;return o;}"
            "Texture2D t:register(t0);SamplerState s:register(s0);"
            "float4 ps(V i):SV_TARGET{return float4(t.Sample(s,i.uv).rgb,1);}";
        ComPtr<ID3DBlob> vs, ps, errors;
        check(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vs, &errors), "compile preview vertex shader");
        check(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &ps, &errors), "compile preview pixel shader");
        check(device->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), nullptr, &vertex_shader), "preview vertex shader");
        check(device->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), nullptr, &pixel_shader), "preview pixel shader");
        D3D11_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
        check(device->CreateSamplerState(&sampler_desc, &sampler), "preview sampler");
        check(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2d.GetAddressOf()), "D2D factory");
        ComPtr<IDXGISurface> surface;
        check(backbuffer.As(&surface), "preview DXGI surface");
        auto props = D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_HARDWARE,
                     D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE), 96, 96);
        check(d2d->CreateDxgiSurfaceRenderTarget(surface.Get(), props, &overlay_target), "GPU overlay surface");
        check(overlay_target->CreateSolidColorBrush(D2D1::ColorF(0,1,0), &brush), "overlay brush");
        check(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(write.GetAddressOf())), "text factory");
        check(write->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
                                     DWRITE_FONT_STRETCH_NORMAL, 14, L"en-us", &font), "overlay font");
    }

    void ensure_texture(DXGI_FORMAT format) {
        if(texture) {
            if(format != texture_format) throw std::runtime_error("Camera GPU texture format changed");
            return;
        }
        if(format != DXGI_FORMAT_B8G8R8A8_UNORM && format != DXGI_FORMAT_B8G8R8X8_UNORM)
            throw std::runtime_error("Camera did not produce a supported BGRA GPU texture");
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width=width; desc.Height=height; desc.MipLevels=1; desc.ArraySize=1;
        desc.Format=format; desc.SampleDesc.Count=1; desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        check(device->CreateTexture2D(&desc, nullptr, &texture), "create CUDA interop texture");
        check(device->CreateShaderResourceView(texture.Get(), nullptr, &texture_view), "preview texture view");
        cuda_check(cudaGraphicsD3D11RegisterResource(&resource, texture.Get(), cudaGraphicsRegisterFlagsNone), "register NVIDIA CUDA/D3D11 texture");
        texture_format=format;
    }

    void read(void* gpu_output, size_t bytes) {
        if(!gpu_output || bytes != size_t(width)*height*4) throw std::runtime_error("Expected a 640x480x4 CUDA output buffer");
        ComPtr<IMFSample> sample;
        DWORD flags=0;
        for(int attempts=0; attempts<100 && !sample; ++attempts) {
            check(reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, &flags, nullptr, &sample), "read camera GPU sample");
            if(flags & MF_SOURCE_READERF_ENDOFSTREAM) throw std::runtime_error("Camera stream ended");
        }
        if(!sample) throw std::runtime_error("Camera returned no video sample");
        ComPtr<IMFMediaBuffer> buffer;
        check(sample->GetBufferByIndex(0, &buffer), "camera sample buffer");
        ComPtr<IMFDXGIBuffer> dxgi;
        if(FAILED(buffer.As(&dxgi))) throw std::runtime_error(
            "Camera backend returned CPU memory instead of a D3D11 surface. Strict GPU video mode has no CPU fallback.");
        ComPtr<ID3D11Texture2D> incoming;
        check(dxgi->GetResource(IID_PPV_ARGS(&incoming)), "camera D3D11 texture");
        UINT subresource=0;
        check(dxgi->GetSubresourceIndex(&subresource), "camera texture subresource");
        D3D11_TEXTURE2D_DESC desc{}; incoming->GetDesc(&desc);
        if(desc.Width < width || desc.Height < height) throw std::runtime_error("Camera GPU texture dimensions are too small");
        ensure_texture(desc.Format);
        ID3D11ShaderResourceView* empty=nullptr;
        context->PSSetShaderResources(0, 1, &empty);
        D3D11_BOX region{0,0,0,width,height,1};
        context->CopySubresourceRegion(texture.Get(),0,0,0,0,incoming.Get(),subresource,&region);
        cuda_check(cudaGraphicsMapResources(1,&resource), "map camera texture to CUDA");
        try {
            cudaArray_t array;
            cuda_check(cudaGraphicsSubResourceGetMappedArray(&array,resource,0,0), "camera CUDA array");
            cuda_check(cudaMemcpy2DFromArray(gpu_output,width*4,array,0,0,width*4,height,cudaMemcpyDeviceToDevice), "copy camera texture on GPU");
        } catch(...) { cudaGraphicsUnmapResources(1,&resource); throw; }
        cuda_check(cudaGraphicsUnmapResources(1,&resource), "unmap camera texture");
        cuda_check(cudaStreamSynchronize(nullptr), "finish camera device copy");
        have_frame=true;
    }

    void upload_test(const void* gpu_input, size_t bytes) {
        if(!gpu_input || bytes != size_t(width)*height*4) throw std::runtime_error("Invalid GPU test buffer");
        ensure_texture(DXGI_FORMAT_B8G8R8A8_UNORM);
        cuda_check(cudaGraphicsMapResources(1,&resource), "map preview test texture");
        try {
            cudaArray_t array;
            cuda_check(cudaGraphicsSubResourceGetMappedArray(&array,resource,0,0), "test CUDA array");
            cuda_check(cudaMemcpy2DToArray(array,0,0,gpu_input,width*4,width*4,height,cudaMemcpyDeviceToDevice), "GPU preview test copy");
        } catch(...) { cudaGraphicsUnmapResources(1,&resource); throw; }
        cuda_check(cudaGraphicsUnmapResources(1,&resource), "unmap preview test texture");
        have_frame=true;
    }

    void present(const Overlay* overlays, int count) {
        if(!have_frame) return;
        if(count<0 || count>8 || (count && !overlays)) throw std::runtime_error("Invalid overlay list");
        ID3D11RenderTargetView* rt=target.Get(); context->OMSetRenderTargets(1,&rt,nullptr);
        D3D11_VIEWPORT viewport{0,0,float(view_width),float(view_height),0,1};
        context->RSSetViewports(1,&viewport);
        context->IASetInputLayout(nullptr); context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex_shader.Get(),nullptr,0); context->PSSetShader(pixel_shader.Get(),nullptr,0);
        ID3D11ShaderResourceView* view=texture_view.Get(); context->PSSetShaderResources(0,1,&view);
        ID3D11SamplerState* state=sampler.Get(); context->PSSetSamplers(0,1,&state);
        context->Draw(3,0);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        overlay_target->BeginDraw();
        overlay_target->SetTransform(D2D1::Matrix3x2F::Scale(float(view_width)/width,float(view_height)/height));
        for(int i=0;i<count;++i) {
            const auto& o=overlays[i];
            brush->SetColor(D2D1::ColorF(o.red,o.green,o.blue));
            overlay_target->DrawRectangle(D2D1::RectF(o.box[0],o.box[1],o.box[2],o.box[3]),brush.Get(),2);
            for(int j=0;j<5;++j) overlay_target->FillEllipse(D2D1::Ellipse(D2D1::Point2F(o.landmarks[j*2],o.landmarks[j*2+1]),2,2),brush.Get());
            wchar_t label[128]{};
            int n=MultiByteToWideChar(CP_UTF8,0,o.label,static_cast<int>(strnlen_s(o.label,128)),label,128);
            overlay_target->DrawText(label,n,font.Get(),D2D1::RectF(std::max(0.f,o.box[0]),std::max(0.f,o.box[1]-22),float(width),o.box[1]+2),brush.Get());
        }
        check(overlay_target->EndDraw(), "GPU overlays");
        check(swap->Present(0,0), "GPU preview presentation");
    }
};
}

VIDEO_API void* flvt_video_create(int camera, void* window, int preview_only, char* error, size_t capacity) noexcept {
    try { auto video=std::make_unique<Video>(); video->open(camera,static_cast<HWND>(window),!preview_only); return video.release(); }
    catch(const std::exception& e) { error_text(error,capacity,e.what()); return nullptr; }
}
VIDEO_API int flvt_video_read(void* handle, void* gpu_output, size_t bytes, char* error, size_t capacity) noexcept {
    try { if(!handle) throw std::runtime_error("Video handle is closed"); static_cast<Video*>(handle)->read(gpu_output,bytes); return 0; }
    catch(const std::exception& e) { error_text(error,capacity,e.what()); return 1; }
}
VIDEO_API int flvt_video_test_frame(void* handle, const void* gpu_input, size_t bytes, char* error, size_t capacity) noexcept {
    try { if(!handle) throw std::runtime_error("Video handle is closed"); static_cast<Video*>(handle)->upload_test(gpu_input,bytes); return 0; }
    catch(const std::exception& e) { error_text(error,capacity,e.what()); return 1; }
}
VIDEO_API int flvt_video_present(void* handle, const Overlay* overlays, int count, char* error, size_t capacity) noexcept {
    try { if(!handle) throw std::runtime_error("Video handle is closed"); static_cast<Video*>(handle)->present(overlays,count); return 0; }
    catch(const std::exception& e) { error_text(error,capacity,e.what()); return 1; }
}
VIDEO_API void flvt_video_destroy(void* handle) noexcept { delete static_cast<Video*>(handle); }
