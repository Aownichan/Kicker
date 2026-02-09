//=================================================================================-----
// SegmentMode Viewer (OptiTrack Camera SDK)
// MODIFIED: publish RGBA frames via Windows shared memory.
// OLD VERSION: OUTDATED
//=================================================================================-----

#include "supportcode.h"
#include "cameralibrary.h"
using namespace CameraLibrary;

#include <windows.h>
#include <cstdint>
#include <chrono>
#include <sddl.h>

static const wchar_t* SHM_NAME = L"Local\\OptiTrackFlex3Frames";

#pragma pack(push, 1)
struct SharedHeader
{
    uint32_t magic;       // 'OTSH' = 0x4853544F
    uint32_t version;     // 1
    uint32_t width;
    uint32_t height;
    uint32_t frame_id;
    double   timestamp_s;
    uint32_t data_size;   // width * height * 4 (RGBA)
};
#pragma pack(pop)

static double NowSeconds()
{
    static auto t0 = std::chrono::steady_clock::now();
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t - t0;
    return dt.count();
}

int main(int argc, char* argv[])
{
    CameraLibrary_EnableDevelopment();
    CameraManager::X();

    PopWaitingDialog();

    std::shared_ptr<Camera> camera = CameraManager::X().GetCamera();
    if (!camera)
    {
        MessageBox(nullptr, "Please connect a camera", "No Device Connected", MB_OK);
        return 1;
    }

    const int cameraWidth = camera->Width();
    const int cameraHeight = camera->Height();

    // Smaller preview window (optional)
    if (!CreateAppWindow(
        "SegmentMode + SharedMemory (RGBA)",
        cameraWidth / 2,
        cameraHeight / 2,
        32,
        gFullscreen))
    {
        return 0;
    }

    // Framebuffer target
    Surface Texture(cameraWidth, cameraHeight);
    Bitmap* framebuffer = new Bitmap(
        cameraWidth,
        cameraHeight,
        Texture.PixelSpan() * 4,
        Bitmap::ThirtyTwoBit,
        Texture.GetBuffer()
    );

    // Segment mode
    camera->SetVideoType(Core::SegmentMode);

    camera->SetTextOverlay(false);

    camera->Start();

    // -------------------------------------------------------------------------
    // Shared memory creation with permissive security (Everyone: RW)
    // -------------------------------------------------------------------------
    const uint32_t w = static_cast<uint32_t>(cameraWidth);
    const uint32_t h = static_cast<uint32_t>(cameraHeight);

    const uint32_t dataSize = w * h * 4; // RGBA
    const size_t shmSize = sizeof(SharedHeader) + static_cast<size_t>(dataSize);

    PSECURITY_DESCRIPTOR pSD = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
        L"D:(A;;GA;;;WD)",
        SDDL_REVISION_1,
        &pSD,
        nullptr))
    {
        MessageBox(nullptr, "Failed to create security descriptor", "Shared Memory Error", MB_OK);
        return 1;
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = pSD;
    sa.bInheritHandle = FALSE;

    HANDLE hMap = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        &sa,
        PAGE_READWRITE,
        0,
        static_cast<DWORD>(shmSize),
        SHM_NAME
    );

    LocalFree(pSD);

    if (!hMap)
    {
        MessageBox(nullptr, "CreateFileMappingW failed", "Shared Memory Error", MB_OK);
        return 1;
    }

    unsigned char* shmBase = static_cast<unsigned char*>(
        MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, shmSize)
        );

    if (!shmBase)
    {
        CloseHandle(hMap);
        MessageBox(nullptr, "MapViewOfFile failed", "Shared Memory Error", MB_OK);
        return 1;
    }

    auto* hdr = reinterpret_cast<SharedHeader*>(shmBase);
    unsigned char* shmRGBA = shmBase + sizeof(SharedHeader);

    hdr->magic = 0x4853544F;
    hdr->version = 1;
    hdr->width = w;
    hdr->height = h;
    hdr->frame_id = 0;
    hdr->timestamp_s = 0.0;
    hdr->data_size = dataSize;

    // -------------------------------------------------------------------------
    // Main loop
    // -------------------------------------------------------------------------
    while (true)
    {
        std::shared_ptr<const Frame> frame = camera->LatestFrame();
        if (frame)
        {
            // Rasterize into Texture buffer (RGBA)
            frame->Rasterize(*camera, framebuffer);

            //frame->Object(0)->X

            const unsigned char* rgba = Texture.GetBuffer();
            const int spanPixels = Texture.PixelSpan();

            // Copy row-by-row because spanPixels can be > width (power-of-2 surface)
            unsigned char* dst = shmRGBA;
            for (int y = 0; y < cameraHeight; y++)
            {
                const unsigned char* srcRow = rgba + (y * spanPixels * 4);
                memcpy(dst + (static_cast<size_t>(y) * cameraWidth * 4),
                    srcRow,
                    static_cast<size_t>(cameraWidth) * 4);
            }

            // Update header last (so reader sees fully written pixels)
            hdr->timestamp_s = NowSeconds();
            hdr->frame_id++;

            if (!DrawGLScene(&Texture))
                break;

            if (keys[VK_ESCAPE])
                break;
        }

        Sleep(1);

        if (!PumpMessages())
            break;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    UnmapViewOfFile(shmBase);
    CloseHandle(hMap);

    CloseWindow();
    CameraManager::X().Shutdown();
    return 0;
}
