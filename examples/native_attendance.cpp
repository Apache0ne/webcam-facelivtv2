#include "facelivt/runtime.hpp"
#include "facelivt/scrfd.hpp"
#include <windows.h>
#include <shellapi.h>
#include <bcrypt.h>
#include <commctrl.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cwchar>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cwctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iomanip>
#include <locale>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "comctl32.lib")

namespace {
constexpr int kFrameWidth=640, kFrameHeight=480, kCropBytes=112*112*3, kEmbedding=512;
constexpr int kEditName=101, kEnroll=102, kCancel=103, kStatus=104, kStats=105, kPeople=106, kBenchmark=107;
constexpr int kCaptureSamples=8;

struct NativeOverlay {
    float box[4], landmarks[10], red, green, blue;
    char label[128];
};
extern "C" __declspec(dllimport) void* flvt_video_create(int,void*,int,char*,size_t);
extern "C" __declspec(dllimport) int flvt_video_read(void*,void*,size_t,char*,size_t);
extern "C" __declspec(dllimport) int flvt_video_present(void*,const NativeOverlay*,int,char*,size_t);
extern "C" __declspec(dllimport) double flvt_video_get_frame_rate(void*);
extern "C" __declspec(dllimport) void flvt_video_destroy(void*);

using Clock=std::chrono::steady_clock;
struct Person { std::string id,created_at; int samples=kCaptureSamples; std::wstring name; std::array<float,kEmbedding> embedding{}; };
struct Attendance { std::wstring name; std::chrono::system_clock::time_point first,last; float similarity=0; };

void cuda_check(cudaError_t code,const char* operation) {
    if(code!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(code));
}

std::wstring utf8_to_wide(const std::string& value) {
    if(value.empty()) return {};
    const int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),nullptr,0);
    if(!n) throw std::runtime_error("Invalid UTF-8 in enrollment database");
    std::wstring result(static_cast<size_t>(n),L'\0');
    MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),result.data(),n);
    return result;
}
std::string wide_to_utf8(const std::wstring& value) {
    if(value.empty()) return {};
    const int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),nullptr,0,nullptr,nullptr);
    if(!n) throw std::runtime_error("Could not encode a name as UTF-8");
    std::string result(static_cast<size_t>(n),'\0');
    WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),result.data(),n,nullptr,nullptr);
    return result;
}
std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for(unsigned char c:value) {
        switch(c) {
        case '"': out<<"\\\""; break;
        case '\\': out<<"\\\\"; break;
        case '\b': out<<"\\b"; break;
        case '\f': out<<"\\f"; break;
        case '\n': out<<"\\n"; break;
        case '\r': out<<"\\r"; break;
        case '\t': out<<"\\t"; break;
        default:
            if(c<0x20) { constexpr char hex[]="0123456789abcdef"; out<<"\\u00"<<hex[c>>4]<<hex[c&15]; }
            else out<<static_cast<char>(c);
        }
    }
    return out.str();
}
std::string json_unescape(const std::string& value) {
    std::string out; out.reserve(value.size());
    auto append_codepoint=[&](unsigned code) {
        if(code<=0x7f) out.push_back(static_cast<char>(code));
        else if(code<=0x7ff) { out.push_back(static_cast<char>(0xc0|(code>>6))); out.push_back(static_cast<char>(0x80|(code&63))); }
        else if(code<=0xffff) { out.push_back(static_cast<char>(0xe0|(code>>12))); out.push_back(static_cast<char>(0x80|((code>>6)&63))); out.push_back(static_cast<char>(0x80|(code&63))); }
        else { out.push_back(static_cast<char>(0xf0|(code>>18))); out.push_back(static_cast<char>(0x80|((code>>12)&63))); out.push_back(static_cast<char>(0x80|((code>>6)&63))); out.push_back(static_cast<char>(0x80|(code&63))); }
    };
    for(size_t i=0;i<value.size();++i) {
        if(value[i]!='\\') { out.push_back(value[i]); continue; }
        if(++i>=value.size()) throw std::runtime_error("Malformed JSON escape in enrollment name");
        switch(value[i]) {
        case '"': out.push_back('"'); break; case '\\': out.push_back('\\'); break; case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break; case 'f': out.push_back('\f'); break; case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break; case 't': out.push_back('\t'); break;
        case 'u': {
            if(i+4>=value.size()) throw std::runtime_error("Incomplete Unicode escape in enrollment name");
            unsigned code=0;
            for(int j=1;j<=4;++j) { char c=value[i+j]; code=(code<<4)|(c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:throw std::runtime_error("Invalid Unicode escape in enrollment name")); }
            i+=4;
            if(code>=0xd800 && code<=0xdbff && i+6<value.size() && value[i+1]=='\\' && value[i+2]=='u') {
                unsigned low=0;
                for(int j=3;j<=6;++j) { char c=value[i+j]; low=(low<<4)|(c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:throw std::runtime_error("Invalid Unicode escape in enrollment name")); }
                if(low>=0xdc00 && low<=0xdfff) { code=0x10000+((code-0xd800)<<10)+(low-0xdc00); i+=6; }
            }
            append_codepoint(code); break;
        }
        default: throw std::runtime_error("Unknown JSON escape in enrollment name");
        }
    }
    return out;
}

std::string sha256_file(const std::filesystem::path& path) {
    BCRYPT_ALG_HANDLE algorithm=nullptr; BCRYPT_HASH_HANDLE hash=nullptr; DWORD object_size=0,returned=0;
    auto check=[](NTSTATUS status,const char* action) { if(status<0) throw std::runtime_error(std::string(action)+" failed (NTSTATUS "+std::to_string(static_cast<unsigned long>(status))+")"); };
    check(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0),"open SHA-256 provider");
    try {
        check(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_size),sizeof(object_size),&returned,0),"query SHA-256 state size");
        std::vector<UCHAR> object(object_size); check(BCryptCreateHash(algorithm,&hash,object.data(),object_size,nullptr,0,0),"create SHA-256 hash");
        std::ifstream file(path,std::ios::binary); if(!file) throw std::runtime_error("Could not open FaceLiVT weights to verify their hash");
        std::array<UCHAR,64*1024> block{};
        while(file) { file.read(reinterpret_cast<char*>(block.data()),static_cast<std::streamsize>(block.size())); const auto n=file.gcount(); if(n>0) check(BCryptHashData(hash,block.data(),static_cast<ULONG>(n),0),"hash FaceLiVT weights"); }
        if(!file.eof()) throw std::runtime_error("Failed while reading FaceLiVT weights for SHA-256");
        std::array<UCHAR,32> digest{}; check(BCryptFinishHash(hash,digest.data(),static_cast<ULONG>(digest.size()),0),"finish SHA-256 hash");
        BCryptDestroyHash(hash); hash=nullptr; BCryptCloseAlgorithmProvider(algorithm,0); algorithm=nullptr;
        constexpr char digits[]="0123456789abcdef"; std::string result; result.reserve(64);
        for(UCHAR byte:digest) { result.push_back(digits[byte>>4]); result.push_back(digits[byte&15]); }
        return result;
    } catch(...) { if(hash) BCryptDestroyHash(hash); if(algorithm) BCryptCloseAlgorithmProvider(algorithm,0); throw; }
}
std::string new_id() {
    std::random_device random; constexpr char digits[]="0123456789abcdef"; std::string id; id.reserve(32);
    for(int i=0;i<16;++i) { const auto byte=static_cast<unsigned>(random())&255u; id.push_back(digits[byte>>4]); id.push_back(digits[byte&15]); }
    return id;
}
std::string iso_utc_now() {
    const auto time=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); std::tm utc{}; gmtime_s(&utc,&time);
    char result[32]{}; std::strftime(result,sizeof(result),"%Y-%m-%dT%H:%M:%SZ",&utc); return result;
}
bool same_name(const std::wstring& a,const std::wstring& b) {
    return CompareStringOrdinal(a.data(),static_cast<int>(a.size()),b.data(),static_cast<int>(b.size()),TRUE)==CSTR_EQUAL;
}

std::vector<Person> load_database(const std::filesystem::path& path,const std::string& model_hash) {
    std::vector<Person> people;
    if(!std::filesystem::exists(path)) return people;
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("Could not read enrollment JSON: "+path.string());
    std::string json((std::istreambuf_iterator<char>(input)),{});
    if(!std::regex_search(json,std::regex(R"rx("schema_version"\s*:\s*1\b)rx"))) throw std::runtime_error("Unsupported enrollment JSON schema version");
    std::smatch hash_match; const std::regex hash_pattern(R"rx("model_sha256"\s*:\s*"([0-9a-fA-F]{64})")rx");
    if(!std::regex_search(json,hash_match,hash_pattern)) throw std::runtime_error("Enrollment JSON has no model SHA-256");
    std::string stored_hash=hash_match[1].str(); std::transform(stored_hash.begin(),stored_hash.end(),stored_hash.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
    if(stored_hash!=model_hash) throw std::runtime_error("Enrollment database was created with a different FaceLiVT checkpoint");
    if(!std::regex_search(json,std::regex(R"rx("embedding_size"\s*:\s*512\b)rx")) ||
       !std::regex_search(json,std::regex(R"rx("alignment"\s*:\s*"five-point-112")rx")))
        throw std::runtime_error("Enrollment JSON has an unsupported embedding size or alignment");
    const std::regex record(R"json(\{[^{}]*"name"\s*:\s*"((?:\\.|[^"\\])*)"[^{}]*"embedding"\s*:\s*\[([^\]]*)\][^{}]*\})json");
    for(std::sregex_iterator it(json.begin(),json.end(),record),end;it!=end;++it) {
        const std::string object=it->str(); Person p; p.name=utf8_to_wide(json_unescape((*it)[1].str()));
        std::smatch field; const std::regex id_pattern(R"rx("id"\s*:\s*"((?:\\.|[^"\\])*)")rx");
        if(std::regex_search(object,field,id_pattern)) p.id=json_unescape(field[1].str());
        const std::regex created_pattern(R"rx("created_at"\s*:\s*"((?:\\.|[^"\\])*)")rx");
        if(std::regex_search(object,field,created_pattern)) p.created_at=json_unescape(field[1].str());
        const std::regex samples_pattern(R"("samples"\s*:\s*(\d+))");
        if(std::regex_search(object,field,samples_pattern)) p.samples=std::stoi(field[1].str());
        if(p.id.empty()) p.id=new_id(); if(p.created_at.empty()) p.created_at=iso_utc_now();
        std::string numbers=(*it)[2].str();
        size_t pos=0;
        for(int i=0;i<kEmbedding;++i) {
            while(pos<numbers.size() && (std::isspace(static_cast<unsigned char>(numbers[pos])) || numbers[pos]==',')) ++pos;
            if(pos==numbers.size()) throw std::runtime_error("Enrollment embedding has fewer than 512 values");
            char* end_number=nullptr;
            p.embedding[i]=std::strtof(numbers.c_str()+pos,&end_number);
            if(end_number==numbers.c_str()+pos || !std::isfinite(p.embedding[i])) throw std::runtime_error("Invalid number in enrollment embedding");
            pos=static_cast<size_t>(end_number-numbers.c_str());
        }
        while(pos<numbers.size() && (std::isspace(static_cast<unsigned char>(numbers[pos])) || numbers[pos]==',')) ++pos;
        if(pos!=numbers.size()) throw std::runtime_error("Enrollment embedding has more than 512 values");
        float norm=0; for(float x:p.embedding) norm+=x*x;
        if(!(norm>0.9f && norm<1.1f)) throw std::runtime_error("Enrollment embedding is not normalized");
        for(const auto& existing:people) if(existing.id==p.id || same_name(existing.name,p.name)) throw std::runtime_error("Enrollment JSON contains duplicate IDs or names");
        people.push_back(std::move(p));
    }
    if(json.find("\"people\"")==std::string::npos) throw std::runtime_error("Enrollment JSON has no people array");
    return people;
}

void save_database(const std::filesystem::path& path,const std::vector<Person>& people,const std::string& model_hash) {
    std::filesystem::create_directories(path.parent_path());
    auto temporary=path; temporary+=L".tmp";
    std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
    if(!out) throw std::runtime_error("Could not write enrollment JSON: "+temporary.string());
    out.imbue(std::locale::classic()); out<<std::setprecision(std::numeric_limits<float>::max_digits10);
    out<<"{\n  \"schema_version\": 1,\n  \"model\": \"facelivtv2_l\",\n  \"model_sha256\": \""<<model_hash
       <<"\",\n  \"embedding_size\": 512,\n  \"alignment\": \"five-point-112\",\n  \"people\": [\n";
    for(size_t i=0;i<people.size();++i) {
        out<<"    {\"id\": \""<<json_escape(people[i].id)<<"\", \"name\": \""<<json_escape(wide_to_utf8(people[i].name))
           <<"\", \"created_at\": \""<<json_escape(people[i].created_at)<<"\", \"samples\": "<<people[i].samples<<", \"embedding\": [";
        for(int j=0;j<kEmbedding;++j) { if(j) out<<", "; out<<people[i].embedding[j]; }
        out<<"]}"<<(i+1<people.size()?",":"")<<"\n";
    }
    out<<"  ]\n}\n"; out.close();
    if(!out) throw std::runtime_error("Failed while writing enrollment JSON");
    if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Could not replace enrollment JSON (Win32 error "+std::to_string(GetLastError())+")");
}

std::wstring clock_text(std::chrono::system_clock::time_point value) {
    const auto t=std::chrono::system_clock::to_time_t(value); std::tm local{}; localtime_s(&local,&t);
    wchar_t text[16]{}; wcsftime(text,std::size(text),L"%H:%M:%S",&local); return text;
}
float cosine(const std::array<float,kEmbedding>& a,const std::array<float,kEmbedding>& b) {
    float score=0; for(int i=0;i<kEmbedding;++i) score+=a[i]*b[i]; return score;
}
std::array<float,kEmbedding> normalize(std::array<float,kEmbedding> value) {
    double norm=0; for(float x:value) norm+=static_cast<double>(x)*x;
    if(norm<1e-12) throw std::runtime_error("FaceLiVT produced an empty embedding");
    const float scale=static_cast<float>(1.0/std::sqrt(norm)); for(float& x:value) x*=scale; return value;
}

class AttendanceApp {
public:
    int run(HINSTANCE instance,int show,const std::filesystem::path& root,bool benchmark_on_start,int camera_index) {
        benchmark_on_start_=benchmark_on_start;
        root_=root; model_path_=root/L"facelivtv2-l.fp16.flvt"; manifest_path_=root/L"generated/kernels/kernel_manifest.tsv";
        detector_path_=root/L"models/scrfd_10g_bnkps.onnx"; database_path_=root/L"data/enrollments.json";
        log_path_=root/L"logs/native-attendance.log";
        std::filesystem::create_directories(log_path_.parent_path());
        log_line("native UI starting");
        const auto init=INITCOMMONCONTROLSEX{sizeof(INITCOMMONCONTROLSEX),ICC_LISTVIEW_CLASSES};
        InitCommonControlsEx(&init);
        WNDCLASSEXW wc{sizeof(wc)}; wc.lpfnWndProc=window_proc; wc.hInstance=instance; wc.hCursor=LoadCursorW(nullptr,IDC_ARROW);
        wc.hbrBackground=reinterpret_cast<HBRUSH>(COLOR_WINDOW+1); wc.lpszClassName=L"FaceLiVTNativeAttendance";
        if(!RegisterClassExW(&wc) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) throw std::runtime_error("Could not register native attendance window");
        hwnd_=CreateWindowExW(0,wc.lpszClassName,L"FaceLiVT | Native CUDA Live Attendance",WS_OVERLAPPEDWINDOW,
                              CW_USEDEFAULT,CW_USEDEFAULT,1320,900,nullptr,nullptr,instance,this);
        if(!hwnd_) throw std::runtime_error("Could not create native attendance window");
        ShowWindow(hwnd_,show); UpdateWindow(hwnd_); layout();
        try {
            set_status(L"Loading native CUDA runtime and enrollment data...");
            model_hash_=sha256_file(model_path_);
            people_=load_database(database_path_,model_hash_);
            if(!std::filesystem::exists(database_path_)) save_database(database_path_,people_,model_hash_);
            log_line("loaded enrollment database: "+std::to_string(people_.size())+" person(s)"); refresh_people();
            facelivt::RuntimeOptions runtime_options; runtime_options.max_batch=8;
            runtime_=std::make_unique<facelivt::Runtime>(model_path_,manifest_path_,runtime_options);
            log_line("FaceLiVT CUDA runtime loaded");
            detector_=std::make_unique<facelivt::ScrfdDetector>(detector_path_);
            log_line("SCRFD CUDA detector loaded");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&frame_),kFrameWidth*kFrameHeight*4),"allocate GPU camera frame");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&crops_),8*kCropBytes),"allocate GPU face crops");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&embeddings_),8*kEmbedding*sizeof(float)),"allocate GPU embeddings");
            video_=flvt_video_create(camera_index,preview_,0,error_,sizeof(error_));
            if(!video_) throw std::runtime_error(error_);
            camera_rate_=flvt_video_get_frame_rate(video_);
            log_line("webcam and D3D11/CUDA preview opened; requested "+std::to_string(camera_rate_)+" FPS mode");
            set_status(L"Camera ready. Enter a name and enroll one person in view.");
            process_frames();
        } catch(const std::exception& e) {
            log_line(std::string("runtime error: ")+e.what());
            MessageBoxW(hwnd_,utf8_to_wide(e.what()).c_str(),L"FaceLiVT runtime error",MB_OK|MB_ICONERROR);
        }
        shutdown();
        log_line("native runtime shutdown");
        return static_cast<int>(message_.wParam);
    }

private:
    static LRESULT CALLBACK window_proc(HWND hwnd,UINT message,WPARAM wparam,LPARAM lparam) {
        auto* self=reinterpret_cast<AttendanceApp*>(GetWindowLongPtrW(hwnd,GWLP_USERDATA));
        if(message==WM_NCCREATE) { self=static_cast<AttendanceApp*>(reinterpret_cast<CREATESTRUCTW*>(lparam)->lpCreateParams); self->hwnd_=hwnd; SetWindowLongPtrW(hwnd,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(self)); }
        if(self) return self->handle(message,wparam,lparam);
        return DefWindowProcW(hwnd,message,wparam,lparam);
    }
    LRESULT handle(UINT message,WPARAM wparam,LPARAM lparam) {
        switch(message) {
        case WM_CREATE: create_controls(); return 0;
        case WM_SIZE: layout(); return 0;
        case WM_COMMAND:
            if(LOWORD(wparam)==kEnroll && HIWORD(wparam)==BN_CLICKED) begin_enrollment();
            if(LOWORD(wparam)==kCancel && HIWORD(wparam)==BN_CLICKED) cancel_enrollment();
            if(LOWORD(wparam)==kBenchmark && HIWORD(wparam)==BN_CLICKED) benchmark_requested_=true;
            return 0;
        case WM_CLOSE: DestroyWindow(hwnd_); return 0;
        case WM_DESTROY: running_=false; PostQuitMessage(0); return 0;
        default: return DefWindowProcW(hwnd_,message,wparam,lparam);
        }
    }
    void create_controls() {
        preview_=CreateWindowExW(0,L"STATIC",L"",WS_CHILD|WS_VISIBLE|WS_CLIPSIBLINGS,0,0,640,480,hwnd_,nullptr,nullptr,nullptr);
        auto label=[&](const wchar_t* text,int x,int y,int w,int h){ return CreateWindowExW(0,L"STATIC",text,WS_CHILD|WS_VISIBLE,x,y,w,h,hwnd_,nullptr,nullptr,nullptr); };
        label(L"Live attendance",24,18,500,38);
        label(L"SCRFD CUDA + GPU alignment + FaceLiVT native CUDA",24,56,700,24);
        enroll_label_=label(L"Enroll a person",0,0,240,28);
        name_=CreateWindowExW(WS_EX_CLIENTEDGE,L"EDIT",L"",WS_CHILD|WS_VISIBLE|ES_AUTOHSCROLL,0,0,360,30,hwnd_,reinterpret_cast<HMENU>(static_cast<INT_PTR>(kEditName)),nullptr,nullptr);
        enroll_=CreateWindowExW(0,L"BUTTON",L"Enroll (8 samples)",WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,0,0,165,34,hwnd_,reinterpret_cast<HMENU>(static_cast<INT_PTR>(kEnroll)),nullptr,nullptr);
        cancel_=CreateWindowExW(0,L"BUTTON",L"Cancel",WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,0,0,100,34,hwnd_,reinterpret_cast<HMENU>(static_cast<INT_PTR>(kCancel)),nullptr,nullptr);
        benchmark_=CreateWindowExW(0,L"BUTTON",L"Benchmark GPU",WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,0,0,160,32,hwnd_,reinterpret_cast<HMENU>(static_cast<INT_PTR>(kBenchmark)),nullptr,nullptr);
        status_=label(L"Starting...",0,0,360,54);
        stats_=label(L"",24,0,700,26);
        people_title_=label(L"Enrolled people",0,0,360,24);
        people_list_=CreateWindowExW(WS_EX_CLIENTEDGE,WC_LISTVIEWW,L"",WS_CHILD|WS_VISIBLE|LVS_REPORT|LVS_SINGLESEL,0,0,360,160,hwnd_,reinterpret_cast<HMENU>(static_cast<INT_PTR>(kPeople)),nullptr,nullptr);
        ListView_SetExtendedListViewStyle(people_list_,LVS_EX_FULLROWSELECT|LVS_EX_DOUBLEBUFFER);
        LVCOLUMNW column{}; column.mask=LVCF_TEXT|LVCF_WIDTH; column.pszText=const_cast<wchar_t*>(L"Person"); column.cx=330; ListView_InsertColumn(people_list_,0,&column);
        attendance_title_=label(L"Today's attendance",24,0,400,28);
        attendance_list_=CreateWindowExW(WS_EX_CLIENTEDGE,WC_LISTVIEWW,L"",WS_CHILD|WS_VISIBLE|LVS_REPORT|LVS_SINGLESEL,0,0,1260,160,hwnd_,nullptr,nullptr,nullptr);
        ListView_SetExtendedListViewStyle(attendance_list_,LVS_EX_FULLROWSELECT|LVS_EX_DOUBLEBUFFER);
        const wchar_t* headers[]={L"Person",L"First seen",L"Last seen",L"Status",L"Best similarity"};
        const int widths[]={360,160,160,220,180};
        for(int i=0;i<5;++i) { LVCOLUMNW col{}; col.mask=LVCF_TEXT|LVCF_WIDTH; col.pszText=const_cast<wchar_t*>(headers[i]); col.cx=widths[i]; ListView_InsertColumn(attendance_list_,i,&col); }
        cancel_enabled(false);
    }
    void layout() {
        if(!hwnd_||!preview_) return;
        RECT client{}; GetClientRect(hwnd_,&client); const int width=client.right,height=client.bottom;
        const int side_x=std::max(700,width-420), preview_w=side_x-42;
        const int available_h=std::max(300,height-300), preview_h=std::min(available_h,static_cast<int>(preview_w*0.75));
        MoveWindow(preview_,24,92,preview_w,preview_h,TRUE);
        MoveWindow(stats_,24,98+preview_h,preview_w,26,TRUE);
        MoveWindow(enroll_label_,side_x,94,360,30,TRUE);
        MoveWindow(name_,side_x,132,360,32,TRUE); MoveWindow(enroll_,side_x,176,165,36,TRUE); MoveWindow(cancel_,side_x+176,176,100,36,TRUE);
        MoveWindow(benchmark_,side_x,222,160,32,TRUE); MoveWindow(status_,side_x,264,360,48,TRUE);
        MoveWindow(people_title_,side_x,318,360,24,TRUE);
        MoveWindow(people_list_,side_x,344,360,std::max(90,height-554),TRUE);
        MoveWindow(attendance_title_,24,125+preview_h,600,30,TRUE);
        const int table_y=160+preview_h; MoveWindow(attendance_list_,24,table_y,width-48,std::max(90,height-table_y-24),TRUE);
    }
    void set_status(const std::wstring& text) { if(status_) SetWindowTextW(status_,text.c_str()); }
    void log_line(const std::string& text) noexcept {
        try { std::ofstream log(log_path_,std::ios::binary|std::ios::app); if(log) { const auto now=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); std::tm local{}; localtime_s(&local,&now); log<<std::put_time(&local,"%H:%M:%S ")<<text<<"\n"; } } catch(...) {}
    }
    void cancel_enabled(bool enabled) { if(cancel_) EnableWindow(cancel_,enabled); if(enroll_) EnableWindow(enroll_,!enabled); }
    void benchmark_pipeline() {
        if(!frame_ready_) { set_status(L"Wait for the first webcam frame before benchmarking."); return; }
        if(enrollment_active_) { set_status(L"Finish or cancel enrollment before benchmarking."); return; }
        set_status(L"Benchmarking CUDA on the current camera frame. Preview pauses briefly.");
        SetWindowTextW(stats_,L"Uncapped GPU benchmark running..."); UpdateWindow(stats_);
        constexpr int warmup=6, iterations=40;
        double detector_seconds=0,pipeline_seconds=0; size_t measured_faces=0;
        for(int iteration=0;iteration<warmup+iterations;++iteration) {
            const auto start=Clock::now();
            auto faces=detector_->detect(frame_,kFrameWidth,kFrameHeight);
            const auto after_detect=Clock::now();
            const int count=static_cast<int>(std::min<size_t>(faces.size(),8));
            if(count==0) throw std::runtime_error("Benchmark needs a face in the camera view");
            detector_->align(frame_,kFrameWidth,kFrameHeight,std::span<const facelivt::ScrfdFace>(faces.data(),count),crops_);
            runtime_->infer_device(crops_,count,embeddings_,true,true,detector_->stream());
            cuda_check(cudaStreamSynchronize(runtime_->stream()),"finish benchmark FaceLiVT inference");
            const auto end=Clock::now();
            if(iteration>=warmup) {
                detector_seconds+=std::chrono::duration<double>(after_detect-start).count();
                pipeline_seconds+=std::chrono::duration<double>(end-start).count();
                measured_faces+=static_cast<size_t>(count);
            }
        }
        benchmark_detector_fps_=iterations/detector_seconds;
        benchmark_pipeline_fps_=iterations/pipeline_seconds;
        benchmark_faces_=static_cast<int>(measured_faces/iterations);
        std::wstring result=L"Uncapped CUDA benchmark complete (replayed camera frame).";
        set_status(result);
        log_line("uncapped benchmark: detector="+std::to_string(benchmark_detector_fps_)+" FPS; detector+alignment+FaceLiVT="+std::to_string(benchmark_pipeline_fps_)+" FPS");
    }
    void begin_enrollment() {
        wchar_t name[256]{}; GetWindowTextW(name_,name,static_cast<int>(std::size(name)));
        enrollment_name_=name; while(!enrollment_name_.empty() && iswspace(enrollment_name_.back())) enrollment_name_.pop_back();
        const auto first=enrollment_name_.find_first_not_of(L" \t\r\n"); if(first==std::wstring::npos) { set_status(L"Enter a name before enrolling."); return; }
        enrollment_name_=enrollment_name_.substr(first); enrollment_samples_.clear(); enrollment_active_=true;
        for(const auto& person:people_) if(same_name(person.name,enrollment_name_)) { enrollment_active_=false; set_status(L"That name is already enrolled. Recognition is already active for them."); return; }
        last_sample_={}; cancel_enabled(true); set_status(L"Enrollment active. Stay alone in view while 8 samples are captured.");
    }
    void cancel_enrollment() { enrollment_active_=false; enrollment_samples_.clear(); cancel_enabled(false); set_status(L"Enrollment canceled."); }
    void refresh_people() {
        if(!people_list_) return; ListView_DeleteAllItems(people_list_);
        for(size_t i=0;i<people_.size();++i) { std::wstring name=people_[i].name; LVITEMW item{}; item.mask=LVIF_TEXT; item.iItem=static_cast<int>(i); item.pszText=name.data(); ListView_InsertItem(people_list_,&item); }
    }
    void update_attendance(const Attendance& row) {
        auto found=std::find_if(attendance_.begin(),attendance_.end(),[&](const auto& a){return a.name==row.name;});
        size_t index;
        if(found==attendance_.end()) { attendance_.push_back(row); index=attendance_.size()-1; }
        else { index=static_cast<size_t>(found-attendance_.begin()); found->last=row.last; found->similarity=std::max(found->similarity,row.similarity); }
        const auto& current=attendance_[index]; wchar_t similarity[16]{}; swprintf_s(similarity,L"%.3f",current.similarity);
        std::wstring first=clock_text(current.first),last=clock_text(current.last),status=L"In view";
        if(index>=static_cast<size_t>(ListView_GetItemCount(attendance_list_))) { std::wstring name=current.name; LVITEMW item{}; item.mask=LVIF_TEXT; item.iItem=static_cast<int>(index); item.pszText=name.data(); ListView_InsertItem(attendance_list_,&item); }
        ListView_SetItemText(attendance_list_,static_cast<int>(index),1,first.data());
        ListView_SetItemText(attendance_list_,static_cast<int>(index),2,last.data());
        ListView_SetItemText(attendance_list_,static_cast<int>(index),3,status.data());
        ListView_SetItemText(attendance_list_,static_cast<int>(index),4,similarity);
    }
    void finish_enrollment() {
        std::array<float,kEmbedding> mean{};
        for(const auto& sample:enrollment_samples_) for(int i=0;i<kEmbedding;++i) mean[i]+=sample[i]/static_cast<float>(enrollment_samples_.size());
        Person person{new_id(),iso_utc_now(),static_cast<int>(enrollment_samples_.size()),enrollment_name_,normalize(mean)};
        people_.push_back(std::move(person));
        save_database(database_path_,people_,model_hash_); refresh_people(); enrollment_active_=false; enrollment_samples_.clear(); cancel_enabled(false);
        set_status(L"Enrollment saved to data\\enrollments.json. Recognition is active.");
    }
    void process_frames() {
        using namespace std::chrono_literals;
        std::array<float,8*kEmbedding> host_embeddings{}; std::vector<NativeOverlay> overlays;
        auto last_rate=Clock::now(); unsigned frames=0; double fps=0; bool rate_started=false;
        while(running_) {
            while(PeekMessageW(&message_,nullptr,0,0,PM_REMOVE)) { if(message_.message==WM_QUIT) { running_=false; break; } TranslateMessage(&message_); DispatchMessageW(&message_); }
            if(!running_) break;
            if(benchmark_on_start_ && frame_ready_) { benchmark_on_start_=false; benchmark_requested_=true; }
            if(benchmark_requested_) {
                benchmark_requested_=false;
                try { benchmark_pipeline(); }
                catch(const std::exception& e) { set_status(L"GPU benchmark failed: "+utf8_to_wide(e.what())); log_line(std::string("benchmark error: ")+e.what()); }
            }
            const auto started=Clock::now();
            if(flvt_video_read(video_,frame_,kFrameWidth*kFrameHeight*4,error_,sizeof(error_))) throw std::runtime_error(error_);
            frame_ready_=true;
            auto faces=detector_->detect(frame_,kFrameWidth,kFrameHeight);
            const int count=static_cast<int>(std::min<size_t>(faces.size(),8)); overlays.clear();
            if(count>0) {
                detector_->align(frame_,kFrameWidth,kFrameHeight,std::span<const facelivt::ScrfdFace>(faces.data(),count),crops_);
                runtime_->infer_device(crops_,count,embeddings_,true,true,detector_->stream());
                cuda_check(cudaStreamSynchronize(runtime_->stream()),"finish native FaceLiVT inference");
                cuda_check(cudaMemcpy(host_embeddings.data(),embeddings_,static_cast<size_t>(count)*kEmbedding*sizeof(float),cudaMemcpyDeviceToHost),"read compact face embeddings");
            }
            const auto now_system=std::chrono::system_clock::now();
            for(int i=0;i<count;++i) {
                std::array<float,kEmbedding> embedding{}; std::copy_n(host_embeddings.data()+i*kEmbedding,kEmbedding,embedding.data());
                std::wstring label=L"Unknown"; float similarity=0; int best=-1;
                if(enrollment_active_ && count==1 && (last_sample_==Clock::time_point{} || started-last_sample_>=120ms)) {
                    enrollment_samples_.push_back(embedding); last_sample_=started;
                    std::wstring progress=L"Capturing enrollment "+std::to_wstring(enrollment_samples_.size())+L" / "+std::to_wstring(kCaptureSamples)+L". Keep one face in view.";
                    set_status(progress);
                    if(enrollment_samples_.size()>=kCaptureSamples) finish_enrollment();
                } else if(!enrollment_active_ && !people_.empty()) {
                    for(size_t p=0;p<people_.size();++p) { const float value=cosine(embedding,people_[p].embedding); if(value>similarity||best<0) { similarity=value; best=static_cast<int>(p); } }
                    if(best>=0 && similarity>=threshold_) { label=people_[best].name; update_attendance(Attendance{label,now_system,now_system,similarity}); }
                }
                NativeOverlay overlay{}; std::copy(faces[i].box.begin(),faces[i].box.end(),overlay.box); std::copy(faces[i].landmarks.begin(),faces[i].landmarks.end(),overlay.landmarks);
                overlay.red=(label==L"Unknown"?1.0f:0.15f); overlay.green=(label==L"Unknown"?0.2f:0.95f); overlay.blue=0.15f;
                std::wstring decorated=label+L"  "; wchar_t score[24]{}; swprintf_s(score,L"%.2f",best>=0?similarity:faces[i].confidence); decorated+=score;
                std::string utf8=wide_to_utf8(decorated); strncpy_s(overlay.label,sizeof(overlay.label),utf8.c_str(),_TRUNCATE); overlays.push_back(overlay);
            }
            if(flvt_video_present(video_,overlays.data(),static_cast<int>(overlays.size()),error_,sizeof(error_))) throw std::runtime_error(error_);
            const auto finished=Clock::now();
            if(!rate_started) { last_rate=finished; rate_started=true; frames=0; }
            else ++frames;
            if(finished-last_rate>=1s) { fps=frames/std::chrono::duration<double>(finished-last_rate).count(); frames=0; last_rate=finished;
                log_line("processing "+std::to_string(fps)+" FPS; faces="+std::to_string(faces.size())); }
            wchar_t stat[256]{};
            if(benchmark_pipeline_fps_>0) swprintf_s(stat,L"Live %.1f FPS (camera)  |  uncapped: SCRFD %.1f / full pipeline %.1f FPS (%d face(s))  |  live faces %zu",fps,benchmark_detector_fps_,benchmark_pipeline_fps_,benchmark_faces_,faces.size());
            else swprintf_s(stat,L"%.1f live FPS (camera; requested %.0f)  |  %zu face(s)  |  SCRFD-10G CUDA  |  FaceLiVT CUDA",fps,camera_rate_,faces.size());
            SetWindowTextW(stats_,stat);
            if(!enrollment_active_ && count==0 && people_.empty()) set_status(L"No one enrolled yet. Enter a name, then click Enroll.");
        }
    }
    void shutdown() noexcept {
        if(video_) { flvt_video_destroy(video_); video_=nullptr; }
        if(embeddings_) cudaFree(embeddings_); embeddings_=nullptr;
        if(crops_) cudaFree(crops_); crops_=nullptr;
        if(frame_) cudaFree(frame_); frame_=nullptr;
        detector_.reset(); runtime_.reset();
    }

    std::filesystem::path root_,model_path_,manifest_path_,detector_path_,database_path_,log_path_; std::string model_hash_;
    HWND hwnd_=nullptr,preview_=nullptr,name_=nullptr,enroll_=nullptr,cancel_=nullptr,benchmark_=nullptr,status_=nullptr,stats_=nullptr,people_list_=nullptr,attendance_list_=nullptr,enroll_label_=nullptr,people_title_=nullptr,attendance_title_=nullptr;
    MSG message_{}; bool running_=true,enrollment_active_=false,benchmark_requested_=false,benchmark_on_start_=false,frame_ready_=false; float threshold_=0.55f;
    double benchmark_detector_fps_=0,benchmark_pipeline_fps_=0,camera_rate_=30; int benchmark_faces_=0;
    std::wstring enrollment_name_; std::vector<std::array<float,kEmbedding>> enrollment_samples_;
    Clock::time_point last_sample_{}; std::vector<Person> people_; std::vector<Attendance> attendance_;
    std::unique_ptr<facelivt::Runtime> runtime_; std::unique_ptr<facelivt::ScrfdDetector> detector_;
    void* video_=nullptr; uint8_t* frame_=nullptr; uint8_t* crops_=nullptr; float* embeddings_=nullptr; char error_[2048]{};
};
}

int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,LPWSTR command_line,int show) {
    try {
        std::array<wchar_t,32768> executable_path{};
        const DWORD executable_length=GetModuleFileNameW(nullptr,executable_path.data(),static_cast<DWORD>(executable_path.size()));
        if(executable_length==0 || executable_length>=executable_path.size())
            throw std::runtime_error("Could not determine the application folder");
        std::filesystem::path root=std::filesystem::path(std::wstring(executable_path.data(),executable_length)).parent_path();
        bool benchmark=false;
        int camera_index=0;
        int count=0; LPWSTR* args=CommandLineToArgvW(GetCommandLineW(),&count);
        if(args) {
            for(int i=1;i<count;++i) {
                if(std::wstring(args[i])==L"--root" && i+1<count) root=args[++i];
                else if(std::wstring(args[i])==L"--camera" && i+1<count) {
                    wchar_t* end=nullptr;
                    const long parsed=std::wcstol(args[++i],&end,10);
                    if(end==args[i] || *end!=L'\0' || parsed<0 || parsed>32)
                        throw std::runtime_error("camera index must be an integer from 0 to 32");
                    camera_index=static_cast<int>(parsed);
                }
                else if(std::wstring(args[i])==L"--benchmark") benchmark=true;
            }
            LocalFree(args);
        }
        AttendanceApp app; return app.run(instance,show,root,benchmark,camera_index);
    } catch(const std::exception& e) {
        MessageBoxW(nullptr,utf8_to_wide(e.what()).c_str(),L"FaceLiVT native runtime",MB_OK|MB_ICONERROR); return 1;
    }
}
