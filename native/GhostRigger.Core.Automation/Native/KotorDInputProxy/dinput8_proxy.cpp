#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dinput.h>
#include <process.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cwctype>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr GUID kGuidSysMouse = {
    0x6F1D2B60, 0xD5A0, 0x11CF, {0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00}
};
constexpr GUID kGuidSysKeyboard = {
    0x6F1D2B61, 0xD5A0, 0x11CF, {0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00}
};

enum class DeviceKind {
    Unknown,
    Mouse,
    Keyboard,
};

struct InjectedEvent {
    DeviceKind kind = DeviceKind::Unknown;
    DWORD offset = 0;
    DWORD data = 0;
    DWORD readyTick = 0;
};

struct InjectedKeyStroke {
    int key = -1;
    int modifier = -1;
    int holdPolls = 0;
    int releasePolls = 2;
};

HMODULE g_realDInput = nullptr;
HMODULE g_proxyModule = nullptr;
std::wstring g_commandPath;
std::wstring g_logPath;
std::wstring g_hostExe;
std::mutex g_mutex;
std::deque<InjectedEvent> g_events;
std::deque<InjectedKeyStroke> g_keyStrokes;
int g_mouseLeftPolls = 0;
LONG g_mouseDeltaX = 0;
LONG g_mouseDeltaY = 0;
HANDLE g_keyboardNotification = nullptr;
HANDLE g_mouseNotification = nullptr;
HWND g_hostWindow = nullptr;
DWORD g_sequence = 1;
HANDLE g_commandStopEvent = nullptr;
INIT_ONCE g_proxyStateOnce = INIT_ONCE_STATIC_INIT;
INIT_ONCE g_commandWatcherOnce = INIT_ONCE_STATIC_INIT;

using DirectInput8CreateFn = HRESULT(WINAPI *)(HINSTANCE, DWORD, REFIID, LPVOID *, LPUNKNOWN);
using DllCanUnloadNowFn = HRESULT(WINAPI *)();
using DllGetClassObjectFn = HRESULT(WINAPI *)(REFCLSID, REFIID, LPVOID *);
using DllRegisterServerFn = HRESULT(WINAPI *)();
using DllUnregisterServerFn = HRESULT(WINAPI *)();
using GetdfDIJoystickFn = LPCDIDATAFORMAT(WINAPI *)();

std::wstring directoryOf(HMODULE module) {
    wchar_t buffer[MAX_PATH] = {};
    GetModuleFileNameW(module, buffer, MAX_PATH);
    std::wstring path(buffer);
    const size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return L".";
    }
    return path.substr(0, slash);
}

std::wstring fileNameOf(const std::wstring &path) {
    const size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return path;
    }
    return path.substr(slash + 1);
}

std::wstring currentProcessPath() {
    wchar_t buffer[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return std::wstring(buffer);
}

std::string utf8(const std::wstring &value) {
    if (value.empty()) {
        return {};
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }
    std::string out(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, out.data(), size, nullptr, nullptr);
    return out;
}

void appendLog(const std::string &line) {
    if (g_logPath.empty()) {
        return;
    }
    std::ofstream out(g_logPath, std::ios::app);
    out << line << "\n";
}

HMODULE realDInput() {
    if (g_realDInput) {
        return g_realDInput;
    }
    wchar_t systemDir[MAX_PATH] = {};
    GetSystemDirectoryW(systemDir, MAX_PATH);
    std::wstring path(systemDir);
    path += L"\\dinput8.dll";
    g_realDInput = LoadLibraryW(path.c_str());
    return g_realDInput;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

int parseInt(const std::string &text, int fallback) {
    if (text.empty()) {
        return fallback;
    }
    try {
        size_t used = 0;
        int base = 10;
        if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
            base = 16;
        }
        int value = std::stoi(text, &used, base);
        return used == text.size() ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

LONG addLongClamped(LONG value, LONG delta) {
    const long long sum = static_cast<long long>(value) + static_cast<long long>(delta);
    const long long minimum = static_cast<long long>(std::numeric_limits<LONG>::min());
    const long long maximum = static_cast<long long>(std::numeric_limits<LONG>::max());
    return static_cast<LONG>(std::clamp(sum, minimum, maximum));
}

void pushEvent(DeviceKind kind, DWORD offset, DWORD data, DWORD delayMilliseconds = 0) {
    g_events.push_back(InjectedEvent{
        kind, offset, data, GetTickCount() + delayMilliseconds
    });
    HANDLE notification = nullptr;
    if (kind == DeviceKind::Keyboard) {
        notification = g_keyboardNotification;
    } else if (kind == DeviceKind::Mouse) {
        notification = g_mouseNotification;
    }
    if (notification) {
        SetEvent(notification);
    }
}

bool eventReady(DWORD now, DWORD readyTick) {
    return static_cast<LONG>(now - readyTick) >= 0;
}

void rememberNotificationLocked(DeviceKind kind, HANDLE notification) {
    if (kind == DeviceKind::Keyboard) {
        g_keyboardNotification = notification;
    } else if (kind == DeviceKind::Mouse) {
        g_mouseNotification = notification;
    }
}

void queueKeyStrokeLocked(int key, int modifier, int polls) {
    InjectedKeyStroke stroke;
    stroke.key = key;
    stroke.modifier = modifier;
    stroke.holdPolls = std::max(1, polls);
    g_keyStrokes.push_back(stroke);
}

bool sendAbsoluteMouseInput(int screenX, int screenY, DWORD buttonFlag) {
    const int virtualLeft = GetSystemMetrics(SM_XVIRTUALSCREEN);
    const int virtualTop = GetSystemMetrics(SM_YVIRTUALSCREEN);
    const int virtualWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    const int virtualHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    if (virtualWidth <= 1 || virtualHeight <= 1
        || screenX < virtualLeft || screenX >= virtualLeft + virtualWidth
        || screenY < virtualTop || screenY >= virtualTop + virtualHeight) {
        return false;
    }
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dx = MulDiv(screenX - virtualLeft, 65535, virtualWidth - 1);
    input.mi.dy = MulDiv(screenY - virtualTop, 65535, virtualHeight - 1);
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        | MOUSEEVENTF_VIRTUALDESK | buttonFlag;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

DWORD scanInputFlags(int scan, bool keyUp) {
    DWORD flags = KEYEVENTF_SCANCODE;
    if ((scan & 0x80) != 0) {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    if (keyUp) {
        flags |= KEYEVENTF_KEYUP;
    }
    return flags;
}

INPUT scanInput(int scan, bool keyUp) {
    INPUT input = {};
    input.type = INPUT_KEYBOARD;
    input.ki.wScan = static_cast<WORD>(scan & 0x7f);
    input.ki.dwFlags = scanInputFlags(scan, keyUp);
    return input;
}

bool sendScanKeyTap(int scan, int holdMilliseconds) {
    INPUT down = scanInput(scan, false);
    if (SendInput(1, &down, sizeof(INPUT)) != 1) {
        return false;
    }
    Sleep(static_cast<DWORD>(std::clamp(holdMilliseconds, 20, 200)));
    INPUT up = scanInput(scan, true);
    return SendInput(1, &up, sizeof(INPUT)) == 1;
}

bool sendScanKeyCombo(int modifier, int scan, int holdMilliseconds) {
    INPUT down[2] = {scanInput(modifier, false), scanInput(scan, false)};
    if (SendInput(2, down, sizeof(INPUT)) != 2) {
        return false;
    }
    Sleep(static_cast<DWORD>(std::clamp(holdMilliseconds, 20, 200)));
    INPUT up[2] = {scanInput(scan, true), scanInput(modifier, true)};
    return SendInput(2, up, sizeof(INPUT)) == 2;
}

bool postScanKeyTap(int scan, int holdMilliseconds) {
    HWND window = GetForegroundWindow();
    DWORD windowProcess = 0;
    if (!window || !GetWindowThreadProcessId(window, &windowProcess)
        || windowProcess != GetCurrentProcessId()) {
        return false;
    }
    const UINT baseScan = static_cast<UINT>(scan & 0x7f);
    const UINT mapScan = baseScan | (((scan & 0x80) != 0) ? 0xe000u : 0u);
    const UINT virtualKey = MapVirtualKeyW(mapScan, MAPVK_VSC_TO_VK_EX);
    if (!virtualKey) {
        return false;
    }
    const LPARAM extended = ((scan & 0x80) != 0) ? (1ll << 24) : 0;
    const LPARAM down = 1 | (static_cast<LPARAM>(baseScan) << 16) | extended;
    const LPARAM up = down | (1ll << 30) | (1ll << 31);
    const bool postedDown = PostMessageW(window, WM_KEYDOWN, virtualKey, down) != FALSE;
    Sleep(static_cast<DWORD>(std::clamp(holdMilliseconds, 20, 200)));
    const bool postedUp = PostMessageW(window, WM_KEYUP, virtualKey, up) != FALSE;
    return postedDown && postedUp;
}

struct HostWindowSearch {
    DWORD processId = 0;
    HWND window = nullptr;
};

BOOL CALLBACK findHostWindow(HWND window, LPARAM parameter) {
    auto *search = reinterpret_cast<HostWindowSearch *>(parameter);
    DWORD processId = 0;
    GetWindowThreadProcessId(window, &processId);
    if (processId == search->processId && IsWindowVisible(window)
        && GetWindow(window, GW_OWNER) == nullptr) {
        search->window = window;
        return FALSE;
    }
    return TRUE;
}

template <typename Device>
HRESULT holdBackgroundAcquireUntilForeground(Device *real, HRESULT result) {
    if (SUCCEEDED(result) || _wcsicmp(g_hostExe.c_str(), L"swkotor.exe") != 0) {
        return result;
    }

    HostWindowSearch search;
    search.processId = GetCurrentProcessId();
    EnumWindows(findHostWindow, reinterpret_cast<LPARAM>(&search));
    if (!search.window) {
        return result;
    }
    g_hostWindow = search.window;

    appendLog("background DirectInput acquire paused until KOTOR regains focus");
    HRESULT reacquired = result;
    while (IsWindow(search.window)) {
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) {
                PostQuitMessage(static_cast<int>(message.wParam));
                return result;
            }
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if (GetForegroundWindow() == search.window) {
            reacquired = real->Acquire();
            if (SUCCEEDED(reacquired)) {
                break;
            }
        }
        MsgWaitForMultipleObjects(0, nullptr, FALSE, 25, QS_ALLINPUT);
    }
    if (!IsWindow(search.window)) {
        return result;
    }
    appendLog(
        "background DirectInput acquire resumed hr="
        + std::to_string(static_cast<long>(reacquired))
    );
    return reacquired;
}

bool activateHostWindowForInput() {
    HostWindowSearch search;
    search.processId = GetCurrentProcessId();
    EnumWindows(findHostWindow, reinterpret_cast<LPARAM>(&search));
    if (!search.window) {
        appendLog("host input activation failed: no visible process window");
        return false;
    }
    g_hostWindow = search.window;

    const DWORD currentThread = GetCurrentThreadId();
    const DWORD targetThread = GetWindowThreadProcessId(search.window, nullptr);
    const HWND priorForeground = GetForegroundWindow();
    const DWORD foregroundThread = priorForeground
        ? GetWindowThreadProcessId(priorForeground, nullptr)
        : 0;
    bool attachedTarget = false;
    bool attachedForeground = false;
    if (targetThread && targetThread != currentThread) {
        attachedTarget = AttachThreadInput(currentThread, targetThread, TRUE) != FALSE;
    }
    if (foregroundThread && foregroundThread != currentThread
        && foregroundThread != targetThread) {
        attachedForeground = AttachThreadInput(
            currentThread, foregroundThread, TRUE
        ) != FALSE;
    }

    ShowWindow(search.window, SW_RESTORE);
    BringWindowToTop(search.window);
    SwitchToThisWindow(search.window, TRUE);
    SetForegroundWindow(search.window);
    SetActiveWindow(search.window);
    SetFocus(search.window);
    Sleep(350);
    const bool active = GetForegroundWindow() == search.window;

    if (attachedForeground) {
        AttachThreadInput(currentThread, foregroundThread, FALSE);
    }
    if (attachedTarget) {
        AttachThreadInput(currentThread, targetThread, FALSE);
    }
    appendLog(
        "host input activation hwnd="
        + std::to_string(reinterpret_cast<uintptr_t>(search.window))
        + " foreground=" + std::to_string(active)
    );
    return active;
}

bool clickForegroundWindowAt(int screenX, int screenY, int holdMilliseconds) {
    HWND window = GetForegroundWindow();
    DWORD windowProcess = 0;
    if (!window || !GetWindowThreadProcessId(window, &windowProcess)
        || windowProcess != GetCurrentProcessId()) {
        appendLog("mouse_click_at refused: foreground window is not the host process");
        return false;
    }

    POINT clientPoint = {screenX, screenY};
    RECT clientRect = {};
    if (!ScreenToClient(window, &clientPoint) || !GetClientRect(window, &clientRect)
        || clientPoint.x < clientRect.left || clientPoint.x >= clientRect.right
        || clientPoint.y < clientRect.top || clientPoint.y >= clientRect.bottom) {
        appendLog("mouse_click_at refused: target is outside the host client area");
        return false;
    }

    RECT priorClip = {};
    if (!GetClipCursor(&priorClip)) {
        appendLog("mouse_click_at refused: GetClipCursor failed");
        return false;
    }

    const int hold = std::clamp(holdMilliseconds, 20, 200);
    const BOOL unclippedDown = ClipCursor(nullptr);
    const BOOL positionedDown = SetCursorPos(screenX, screenY);
    const bool buttonDown = sendAbsoluteMouseInput(screenX, screenY, MOUSEEVENTF_LEFTDOWN);
    Sleep(static_cast<DWORD>(hold));

    DWORD liveProcess = 0;
    const bool stillForeground = GetForegroundWindow() == window
        && GetWindowThreadProcessId(window, &liveProcess)
        && liveProcess == GetCurrentProcessId();
    BOOL unclippedUp = FALSE;
    BOOL positionedUp = FALSE;
    if (stillForeground) {
        unclippedUp = ClipCursor(nullptr);
        positionedUp = SetCursorPos(screenX, screenY);
    }
    const bool buttonUp = sendAbsoluteMouseInput(screenX, screenY, MOUSEEVENTF_LEFTUP);
    Sleep(30);
    if (stillForeground) {
        ClipCursor(&priorClip);
    } else {
        ClipCursor(nullptr);
    }

    appendLog(
        "mouse_click_at " + std::to_string(screenX) + " " + std::to_string(screenY)
        + " hold=" + std::to_string(hold)
        + " unclip_down=" + std::to_string(unclippedDown != FALSE)
        + " position_down=" + std::to_string(positionedDown != FALSE)
        + " down=" + std::to_string(buttonDown)
        + " unclip_up=" + std::to_string(unclippedUp != FALSE)
        + " position_up=" + std::to_string(positionedUp != FALSE)
        + " up=" + std::to_string(buttonUp)
        + " foreground=" + std::to_string(stillForeground)
    );
    return positionedDown != FALSE && buttonDown && buttonUp;
}

void parseCommandLine(const std::string &line) {
    std::istringstream input(line);
    std::string command;
    input >> command;
    command = lower(command);
    if (command.empty() || command[0] == '#') {
        return;
    }
    if (command == "mouse_click") {
        std::string pollsText;
        input >> pollsText;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_mouseLeftPolls = std::max(g_mouseLeftPolls, parseInt(pollsText, 24));
            pushEvent(DeviceKind::Mouse, DIMOFS_BUTTON0, 0x80);
            pushEvent(DeviceKind::Mouse, DIMOFS_BUTTON0, 0x00);
        }
        appendLog("queued mouse_click");
        return;
    }
    if (command == "mouse_click_at") {
        std::string xText;
        std::string yText;
        std::string holdText;
        if (!(input >> xText >> yText)) {
            appendLog("ignored malformed mouse_click_at");
            return;
        }
        input >> holdText;
        clickForegroundWindowAt(
            parseInt(xText, 0), parseInt(yText, 0), parseInt(holdText, 50)
        );
        return;
    }
    if (command == "mouse_post_at") {
        std::string xText;
        std::string yText;
        if (!(input >> xText >> yText)) {
            appendLog("ignored malformed mouse_post_at");
            return;
        }
        const int screenX = parseInt(xText, 0);
        const int screenY = parseInt(yText, 0);
        const bool foreground = activateHostWindowForInput();
        HWND window = GetForegroundWindow();
        DWORD windowProcess = 0;
        POINT clientPoint = {screenX, screenY};
        const bool valid = foreground && window
            && GetWindowThreadProcessId(window, &windowProcess)
            && windowProcess == GetCurrentProcessId()
            && ScreenToClient(window, &clientPoint) != FALSE;
        bool posted = false;
        if (valid) {
            const LPARAM point = MAKELPARAM(clientPoint.x, clientPoint.y);
            const bool moved = PostMessageW(
                window, WM_MOUSEMOVE, 0, point) != FALSE;
            const bool down = PostMessageW(
                window, WM_LBUTTONDOWN, MK_LBUTTON, point) != FALSE;
            Sleep(40);
            const bool up = PostMessageW(
                window, WM_LBUTTONUP, 0, point) != FALSE;
            posted = moved && down && up;
        }
        appendLog(
            "mouse_post_at " + std::to_string(screenX) + " "
            + std::to_string(screenY)
            + " foreground=" + std::to_string(foreground)
            + " posted=" + std::to_string(posted)
        );
        return;
    }
    if (command == "mouse_move") {
        std::string xText;
        std::string yText;
        if (!(input >> xText >> yText)) {
            appendLog("ignored malformed mouse_move");
            return;
        }
        const int deltaX = parseInt(xText, 0);
        const int deltaY = parseInt(yText, 0);
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_mouseDeltaX = addLongClamped(g_mouseDeltaX, static_cast<LONG>(deltaX));
            g_mouseDeltaY = addLongClamped(g_mouseDeltaY, static_cast<LONG>(deltaY));
        }
        appendLog(
            "queued mouse_move " + std::to_string(deltaX) + " " + std::to_string(deltaY)
        );
        return;
    }
    if (command == "key_tap") {
        std::string keyText;
        std::string pollsText;
        input >> keyText >> pollsText;
        int key = parseInt(keyText, -1);
        if (key >= 0 && key < 256) {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                queueKeyStrokeLocked(key, -1, parseInt(pollsText, 12));
            }
            appendLog("queued key_tap");
        }
        return;
    }
    if (command == "key_send") {
        std::string keyText;
        std::string holdText;
        input >> keyText >> holdText;
        const int key = parseInt(keyText, -1);
        if (key >= 0 && key < 256) {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                queueKeyStrokeLocked(key, -1, 6);
                pushEvent(DeviceKind::Keyboard, static_cast<DWORD>(key), 0x80);
                pushEvent(DeviceKind::Keyboard, static_cast<DWORD>(key), 0x00);
            }
            const bool sent = sendScanKeyTap(key, parseInt(holdText, 50));
            appendLog("key_send sent=" + std::to_string(sent));
        }
        return;
    }
    if (command == "key_buffer") {
        std::string keyText;
        std::string holdText;
        input >> keyText >> holdText;
        const int key = parseInt(keyText, -1);
        const DWORD hold = static_cast<DWORD>(std::clamp(
            parseInt(holdText, 80), 35, 500
        ));
        if (key >= 0 && key < 256) {
            const bool foreground = activateHostWindowForInput();
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                pushEvent(DeviceKind::Keyboard, static_cast<DWORD>(key), 0x80);
                pushEvent(
                    DeviceKind::Keyboard, static_cast<DWORD>(key), 0x00, hold
                );
            }
            const bool wokeDown = g_hostWindow
                && PostMessageW(g_hostWindow, WM_NULL, 0, 0) != FALSE;
            Sleep(hold + 20);
            const bool wokeUp = g_hostWindow
                && PostMessageW(g_hostWindow, WM_NULL, 0, 0) != FALSE;
            appendLog(
                "queued key_buffer key=" + std::to_string(key)
                + " hold=" + std::to_string(hold)
                + " foreground=" + std::to_string(foreground)
                + " wake_down=" + std::to_string(wokeDown)
                + " wake_up=" + std::to_string(wokeUp)
            );
        }
        return;
    }
    if (command == "key_post") {
        std::string keyText;
        std::string holdText;
        input >> keyText >> holdText;
        const int key = parseInt(keyText, -1);
        if (key >= 0 && key < 256) {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                queueKeyStrokeLocked(key, -1, 6);
                pushEvent(DeviceKind::Keyboard, static_cast<DWORD>(key), 0x80);
                pushEvent(DeviceKind::Keyboard, static_cast<DWORD>(key), 0x00);
            }
            const bool posted = postScanKeyTap(key, parseInt(holdText, 50));
            appendLog("key_post posted=" + std::to_string(posted));
        }
        return;
    }
    if (command == "key_message") {
        std::string keyText;
        std::string holdText;
        input >> keyText >> holdText;
        const int key = parseInt(keyText, -1);
        if (key >= 0 && key < 256) {
            const bool foreground = activateHostWindowForInput();
            const bool posted = postScanKeyTap(
                key, parseInt(holdText, 35)
            );
            appendLog(
                "key_message key=" + std::to_string(key)
                + " foreground=" + std::to_string(foreground)
                + " posted=" + std::to_string(posted)
            );
        }
        return;
    }
    if (command == "key_system") {
        std::string keyText;
        std::string holdText;
        input >> keyText >> holdText;
        const int key = parseInt(keyText, -1);
        if (key >= 0 && key < 256) {
            const bool foreground = activateHostWindowForInput();
            const bool sent = sendScanKeyTap(
                key, parseInt(holdText, 35)
            );
            appendLog(
                "key_system key=" + std::to_string(key)
                + " foreground=" + std::to_string(foreground)
                + " sent=" + std::to_string(sent)
            );
        }
        return;
    }
    if (command == "key_combo") {
        std::string modifierText;
        std::string keyText;
        std::string pollsText;
        input >> modifierText >> keyText >> pollsText;
        int modifier = parseInt(modifierText, -1);
        int key = parseInt(keyText, -1);
        int polls = parseInt(pollsText, 12);
        if (modifier >= 0 && modifier < 256 && key >= 0 && key < 256) {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                queueKeyStrokeLocked(key, modifier, polls);
            }
            appendLog("queued key_combo");
        }
        return;
    }
    if (command == "key_combo_send") {
        std::string modifierText;
        std::string keyText;
        std::string holdText;
        input >> modifierText >> keyText >> holdText;
        const int modifier = parseInt(modifierText, -1);
        const int key = parseInt(keyText, -1);
        if (modifier >= 0 && modifier < 256 && key >= 0 && key < 256) {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                queueKeyStrokeLocked(key, modifier, 6);
                pushEvent(
                    DeviceKind::Keyboard, static_cast<DWORD>(modifier), 0x80
                );
                pushEvent(DeviceKind::Keyboard, static_cast<DWORD>(key), 0x80);
                pushEvent(DeviceKind::Keyboard, static_cast<DWORD>(key), 0x00);
                pushEvent(
                    DeviceKind::Keyboard, static_cast<DWORD>(modifier), 0x00
                );
            }
            const bool sent = sendScanKeyCombo(
                modifier, key, parseInt(holdText, 50)
            );
            appendLog("key_combo_send sent=" + std::to_string(sent));
        }
        return;
    }
    if (command == "reset") {
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_mouseLeftPolls = 0;
            g_mouseDeltaX = 0;
            g_mouseDeltaY = 0;
            g_keyStrokes.clear();
            g_events.clear();
        }
        appendLog("reset");
    }
}

void loadCommands() {
    std::vector<std::string> lines;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_commandPath.empty()
            || GetFileAttributesW(g_commandPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
            return;
        }
        std::ifstream in(g_commandPath);
        std::string line;
        while (std::getline(in, line)) {
            lines.push_back(line);
        }
        in.close();
        DeleteFileW(g_commandPath.c_str());
    }
    for (const std::string &item : lines) {
        parseCommandLine(item);
    }
}

unsigned __stdcall commandWatcherThread(void *) {
    appendLog("command watcher started");
    while (g_commandStopEvent) {
        const DWORD waitResult = WaitForSingleObject(g_commandStopEvent, 10);
        if (waitResult == WAIT_OBJECT_0 || waitResult == WAIT_FAILED) {
            break;
        }
        loadCommands();
    }
    appendLog("command watcher stopped");
    return 0U;
}

BOOL CALLBACK startCommandWatcher(PINIT_ONCE, PVOID, PVOID *) {
    g_commandStopEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_commandStopEvent) {
        appendLog("command watcher failed to create stop event");
        return TRUE;
    }
    const uintptr_t thread = _beginthreadex(
        nullptr, 0, commandWatcherThread, nullptr, 0, nullptr
    );
    if (!thread) {
        appendLog("command watcher failed to create thread");
        CloseHandle(g_commandStopEvent);
        g_commandStopEvent = nullptr;
    } else {
        CloseHandle(reinterpret_cast<HANDLE>(thread));
    }
    return TRUE;
}

void ensureCommandWatcher() {
    InitOnceExecuteOnce(&g_commandWatcherOnce, startCommandWatcher, nullptr, nullptr);
}

BOOL CALLBACK initializeProxyState(PINIT_ONCE, PVOID, PVOID *) {
    std::wstring dir = directoryOf(g_proxyModule);
    g_commandPath = dir + L"\\kotor_dinput_proxy_commands.txt";
    g_logPath = dir + L"\\kotor_dinput_proxy.log";
    g_hostExe = fileNameOf(currentProcessPath());

    wchar_t modulePath[MAX_PATH] = {};
    HMODULE pinnedModule = nullptr;
    if (GetModuleFileNameW(g_proxyModule, modulePath, MAX_PATH)) {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, modulePath, &pinnedModule);
    }
    appendLog("loaded kotor dinput proxy for " + utf8(g_hostExe));
    return TRUE;
}

void ensureProxyState() {
    InitOnceExecuteOnce(&g_proxyStateOnce, initializeProxyState, nullptr, nullptr);
}

void applyKeyboardState(DWORD cbData, LPVOID data) {
    if (cbData < 256 || !data) {
        return;
    }
    auto *keys = static_cast<unsigned char *>(data);
    std::lock_guard<std::mutex> lock(g_mutex);

    while (!g_keyStrokes.empty()) {
        InjectedKeyStroke &stroke = g_keyStrokes.front();
        if (stroke.holdPolls > 0) {
            keys[stroke.key] |= 0x80;
            if (stroke.modifier >= 0 && stroke.modifier < 256) {
                keys[stroke.modifier] |= 0x80;
            }
            --stroke.holdPolls;
            return;
        }
        if (stroke.releasePolls > 0) {
            --stroke.releasePolls;
            return;
        }
        g_keyStrokes.pop_front();
    }
}

void applyMouseState(DWORD cbData, LPVOID data) {
    if (cbData < sizeof(DIMOUSESTATE) || !data) {
        return;
    }
    auto *mouse = static_cast<DIMOUSESTATE *>(data);
    std::lock_guard<std::mutex> lock(g_mutex);
    // KOTOR reads unbuffered relative mouse state. Apply each queued movement
    // exactly once here; do not also enqueue axis events or the cursor can move
    // twice if a caller consumes both DirectInput APIs.
    mouse->lX = addLongClamped(mouse->lX, g_mouseDeltaX);
    mouse->lY = addLongClamped(mouse->lY, g_mouseDeltaY);
    g_mouseDeltaX = 0;
    g_mouseDeltaY = 0;
    if (g_mouseLeftPolls > 0) {
        mouse->rgbButtons[0] |= 0x80;
        --g_mouseLeftPolls;
    }
}

bool hasPendingState(DeviceKind kind) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (kind == DeviceKind::Keyboard) {
        return !g_keyStrokes.empty();
    }
    if (kind == DeviceKind::Mouse) {
        return g_mouseDeltaX != 0 || g_mouseDeltaY != 0 || g_mouseLeftPolls > 0;
    }
    return false;
}

bool hasPendingData(DeviceKind kind) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return std::any_of(
        g_events.begin(), g_events.end(),
        [kind](const InjectedEvent &event) { return event.kind == kind; }
    );
}

void appendInjectedData(
    DeviceKind kind,
    DWORD cbObjectData,
    LPDIDEVICEOBJECTDATA data,
    DWORD capacity,
    LPDWORD count,
    DWORD flags
) {
    if (!count || !data || cbObjectData < sizeof(DIDEVICEOBJECTDATA)) {
        return;
    }
    const DWORD initial = *count;
    DWORD used = initial;
    std::vector<std::string> delivered;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (DWORD index = 0; index < initial; ++index) {
            DIDEVICEOBJECTDATA *slot = reinterpret_cast<DIDEVICEOBJECTDATA *>(
                reinterpret_cast<unsigned char *>(data) + (index * cbObjectData)
            );
            if (slot->dwSequence >= g_sequence) {
                g_sequence = slot->dwSequence + 1;
            }
        }
        const DWORD now = GetTickCount();
        auto it = g_events.begin();
        while (used < capacity && it != g_events.end()) {
            if (it->kind != kind) {
                ++it;
                continue;
            }
            if (!eventReady(now, it->readyTick)) {
                // Preserve FIFO ordering for a device.  In particular, never
                // deliver a later key-down ahead of an earlier key-up.
                break;
            }
            DIDEVICEOBJECTDATA *slot = reinterpret_cast<DIDEVICEOBJECTDATA *>(
                reinterpret_cast<unsigned char *>(data) + (used * cbObjectData)
            );
            slot->dwOfs = it->offset;
            slot->dwData = it->data;
            slot->dwTimeStamp = GetTickCount();
            slot->dwSequence = g_sequence++;
            slot->uAppData = 0;
            delivered.push_back(
                "delivered buffered event ofs=" + std::to_string(slot->dwOfs)
                + " data=" + std::to_string(slot->dwData)
                + " sequence=" + std::to_string(slot->dwSequence)
            );
            ++used;
            if ((flags & DIGDD_PEEK) != 0) {
                ++it;
            } else {
                it = g_events.erase(it);
            }
            if (kind == DeviceKind::Keyboard) {
                // KOTOR folds buffered key records into its per-frame input
                // state after GetEvents.  Keep opposite edges in separate
                // polls, even when a menu focus/acquire flush delayed the first
                // read until both original wall-clock deadlines had elapsed.
                if ((slot->dwData & 0x80) != 0 && (flags & DIGDD_PEEK) == 0) {
                    for (auto pending = it; pending != g_events.end(); ++pending) {
                        if (pending->kind == DeviceKind::Keyboard) {
                            pending->readyTick = now + 35;
                            break;
                        }
                    }
                }
                break;
            }
        }
        *count = used;
    }
    for (const std::string &line : delivered) {
        appendLog(line);
    }
    if (used > initial) {
        appendLog(
            "delivered buffered events=" + std::to_string(used - initial)
            + " kind=" + std::to_string(static_cast<int>(kind))
        );
    }
}

class ProxyDeviceA final : public IDirectInputDevice8A {
public:
    ProxyDeviceA(IDirectInputDevice8A *real, DeviceKind kind) : real_(real), kind_(kind) {}

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, LPVOID *out) override {
        if (!out) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown)
            || IsEqualIID(riid, IID_IDirectInputDeviceA)
            || IsEqualIID(riid, IID_IDirectInputDevice2A)
            || IsEqualIID(riid, IID_IDirectInputDevice7A)
            || IsEqualIID(riid, IID_IDirectInputDevice8A)) {
            *out = static_cast<IDirectInputDevice8A *>(this);
            AddRef();
            appendLog("kept wrapped DirectInput device A interface");
            return S_OK;
        }
        return real_->QueryInterface(riid, out);
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&refs_); }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG refs = InterlockedDecrement(&refs_);
        if (!refs) {
            real_->Release();
            delete this;
        }
        return refs;
    }
    HRESULT STDMETHODCALLTYPE GetCapabilities(LPDIDEVCAPS value) override { return real_->GetCapabilities(value); }
    HRESULT STDMETHODCALLTYPE EnumObjects(LPDIENUMDEVICEOBJECTSCALLBACKA cb, LPVOID ref, DWORD flags) override { return real_->EnumObjects(cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE GetProperty(REFGUID prop, LPDIPROPHEADER header) override { return real_->GetProperty(prop, header); }
    HRESULT STDMETHODCALLTYPE SetProperty(REFGUID prop, LPCDIPROPHEADER header) override { return real_->SetProperty(prop, header); }
    HRESULT STDMETHODCALLTYPE Acquire() override {
        const HRESULT hr = holdBackgroundAcquireUntilForeground(
            real_, real_->Acquire()
        );
        const DWORD call = ++acquireCalls_;
        // KOTOR retries Acquire every frame while its window is unfocused.
        // Opening and flushing the log file on each retry can produce
        // megabytes of synchronous I/O and make the game appear hung. Keep the
        // first three observations and subsequent state transitions only.
        if (call <= 3 || hr != lastAcquireResult_) {
            appendLog(
                "Acquire A kind=" + std::to_string(static_cast<int>(kind_))
                + " call=" + std::to_string(call)
                + " hr=" + std::to_string(static_cast<long>(hr))
            );
        }
        lastAcquireResult_ = hr;
        return hr;
    }
    HRESULT STDMETHODCALLTYPE Unacquire() override { return real_->Unacquire(); }
    HRESULT STDMETHODCALLTYPE GetDeviceState(DWORD cbData, LPVOID data) override {
        loadCommands();
        const bool pending = hasPendingState(kind_);
        const DWORD call = ++stateCalls_;
        HRESULT hr = real_->GetDeviceState(cbData, data);
        if (SUCCEEDED(hr)) {
            if (kind_ == DeviceKind::Keyboard) {
                applyKeyboardState(cbData, data);
            } else if (kind_ == DeviceKind::Mouse) {
                applyMouseState(cbData, data);
            }
        }
        if (call <= 3 || pending) {
            appendLog(
                "GetDeviceState A kind="
                + std::to_string(static_cast<int>(kind_))
                + " call=" + std::to_string(call)
                + " cb=" + std::to_string(cbData)
                + " pending=" + std::to_string(pending)
                + " hr=" + std::to_string(static_cast<long>(hr))
            );
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE GetDeviceData(DWORD cbObjectData, LPDIDEVICEOBJECTDATA data, LPDWORD count, DWORD flags) override {
        loadCommands();
        const bool pending = hasPendingData(kind_);
        const DWORD call = ++dataCalls_;
        DWORD requested = count ? *count : 0;
        HRESULT hr = real_->GetDeviceData(cbObjectData, data, count, flags);
        if (count && data && SUCCEEDED(hr)) {
            if (*count < requested) {
                appendInjectedData(
                    kind_, cbObjectData, data, requested, count, flags
                );
            }
        }
        if (call <= 3 || (pending && count && *count > 0)) {
            appendLog(
                "GetDeviceData A kind="
                + std::to_string(static_cast<int>(kind_))
                + " call=" + std::to_string(call)
                + " cb=" + std::to_string(cbObjectData)
                + " requested=" + std::to_string(requested)
                + " returned=" + std::to_string(count ? *count : 0)
                + " flags=" + std::to_string(flags)
                + " pending=" + std::to_string(pending)
                + " hr=" + std::to_string(static_cast<long>(hr))
            );
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE SetDataFormat(LPCDIDATAFORMAT format) override {
        appendLog(
            "SetDataFormat A kind=" + std::to_string(static_cast<int>(kind_))
            + " data_size=" + std::to_string(format ? format->dwDataSize : 0)
            + " objects=" + std::to_string(format ? format->dwNumObjs : 0)
        );
        return real_->SetDataFormat(format);
    }
    HRESULT STDMETHODCALLTYPE SetEventNotification(HANDLE event) override {
        const HRESULT hr = real_->SetEventNotification(event);
        if (SUCCEEDED(hr)) {
            std::lock_guard<std::mutex> lock(g_mutex);
            rememberNotificationLocked(kind_, event);
        }
        appendLog(
            "SetEventNotification A kind="
            + std::to_string(static_cast<int>(kind_))
            + " active=" + std::to_string(event != nullptr)
        );
        return hr;
    }
    HRESULT STDMETHODCALLTYPE SetCooperativeLevel(HWND hwnd, DWORD flags) override { return real_->SetCooperativeLevel(hwnd, flags); }
    HRESULT STDMETHODCALLTYPE GetObjectInfo(LPDIDEVICEOBJECTINSTANCEA out, DWORD object, DWORD how) override { return real_->GetObjectInfo(out, object, how); }
    HRESULT STDMETHODCALLTYPE GetDeviceInfo(LPDIDEVICEINSTANCEA out) override { return real_->GetDeviceInfo(out); }
    HRESULT STDMETHODCALLTYPE RunControlPanel(HWND owner, DWORD flags) override { return real_->RunControlPanel(owner, flags); }
    HRESULT STDMETHODCALLTYPE Initialize(HINSTANCE inst, DWORD version, REFGUID guid) override { return real_->Initialize(inst, version, guid); }
    HRESULT STDMETHODCALLTYPE CreateEffect(REFGUID guid, LPCDIEFFECT effect, LPDIRECTINPUTEFFECT *out, LPUNKNOWN outer) override { return real_->CreateEffect(guid, effect, out, outer); }
    HRESULT STDMETHODCALLTYPE EnumEffects(LPDIENUMEFFECTSCALLBACKA cb, LPVOID ref, DWORD type) override { return real_->EnumEffects(cb, ref, type); }
    HRESULT STDMETHODCALLTYPE GetEffectInfo(LPDIEFFECTINFOA out, REFGUID guid) override { return real_->GetEffectInfo(out, guid); }
    HRESULT STDMETHODCALLTYPE GetForceFeedbackState(LPDWORD out) override { return real_->GetForceFeedbackState(out); }
    HRESULT STDMETHODCALLTYPE SendForceFeedbackCommand(DWORD flags) override { return real_->SendForceFeedbackCommand(flags); }
    HRESULT STDMETHODCALLTYPE EnumCreatedEffectObjects(LPDIENUMCREATEDEFFECTOBJECTSCALLBACK cb, LPVOID ref, DWORD flags) override { return real_->EnumCreatedEffectObjects(cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE Escape(LPDIEFFESCAPE escape) override { return real_->Escape(escape); }
    HRESULT STDMETHODCALLTYPE Poll() override { return real_->Poll(); }
    HRESULT STDMETHODCALLTYPE SendDeviceData(DWORD cbObjectData, LPCDIDEVICEOBJECTDATA data, LPDWORD count, DWORD flags) override { return real_->SendDeviceData(cbObjectData, data, count, flags); }
    HRESULT STDMETHODCALLTYPE EnumEffectsInFile(LPCSTR file, LPDIENUMEFFECTSINFILECALLBACK cb, LPVOID ref, DWORD flags) override { return real_->EnumEffectsInFile(file, cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE WriteEffectToFile(LPCSTR file, DWORD count, LPDIFILEEFFECT effects, DWORD flags) override { return real_->WriteEffectToFile(file, count, effects, flags); }
    HRESULT STDMETHODCALLTYPE BuildActionMap(LPDIACTIONFORMATA format, LPCSTR user, DWORD flags) override { return real_->BuildActionMap(format, user, flags); }
    HRESULT STDMETHODCALLTYPE SetActionMap(LPDIACTIONFORMATA format, LPCSTR user, DWORD flags) override { return real_->SetActionMap(format, user, flags); }
    HRESULT STDMETHODCALLTYPE GetImageInfo(LPDIDEVICEIMAGEINFOHEADERA header) override { return real_->GetImageInfo(header); }

private:
    IDirectInputDevice8A *real_ = nullptr;
    DeviceKind kind_ = DeviceKind::Unknown;
    volatile LONG refs_ = 1;
    DWORD stateCalls_ = 0;
    DWORD dataCalls_ = 0;
    DWORD acquireCalls_ = 0;
    HRESULT lastAcquireResult_ = S_OK;
};

class ProxyDeviceW final : public IDirectInputDevice8W {
public:
    ProxyDeviceW(IDirectInputDevice8W *real, DeviceKind kind) : real_(real), kind_(kind) {}

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, LPVOID *out) override {
        if (!out) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown)
            || IsEqualIID(riid, IID_IDirectInputDeviceW)
            || IsEqualIID(riid, IID_IDirectInputDevice2W)
            || IsEqualIID(riid, IID_IDirectInputDevice7W)
            || IsEqualIID(riid, IID_IDirectInputDevice8W)) {
            *out = static_cast<IDirectInputDevice8W *>(this);
            AddRef();
            appendLog("kept wrapped DirectInput device W interface");
            return S_OK;
        }
        return real_->QueryInterface(riid, out);
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&refs_); }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG refs = InterlockedDecrement(&refs_);
        if (!refs) {
            real_->Release();
            delete this;
        }
        return refs;
    }
    HRESULT STDMETHODCALLTYPE GetCapabilities(LPDIDEVCAPS value) override { return real_->GetCapabilities(value); }
    HRESULT STDMETHODCALLTYPE EnumObjects(LPDIENUMDEVICEOBJECTSCALLBACKW cb, LPVOID ref, DWORD flags) override { return real_->EnumObjects(cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE GetProperty(REFGUID prop, LPDIPROPHEADER header) override { return real_->GetProperty(prop, header); }
    HRESULT STDMETHODCALLTYPE SetProperty(REFGUID prop, LPCDIPROPHEADER header) override { return real_->SetProperty(prop, header); }
    HRESULT STDMETHODCALLTYPE Acquire() override {
        return holdBackgroundAcquireUntilForeground(real_, real_->Acquire());
    }
    HRESULT STDMETHODCALLTYPE Unacquire() override { return real_->Unacquire(); }
    HRESULT STDMETHODCALLTYPE GetDeviceState(DWORD cbData, LPVOID data) override {
        loadCommands();
        HRESULT hr = real_->GetDeviceState(cbData, data);
        if (SUCCEEDED(hr)) {
            if (kind_ == DeviceKind::Keyboard) {
                applyKeyboardState(cbData, data);
            } else if (kind_ == DeviceKind::Mouse) {
                applyMouseState(cbData, data);
            }
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE GetDeviceData(DWORD cbObjectData, LPDIDEVICEOBJECTDATA data, LPDWORD count, DWORD flags) override {
        loadCommands();
        DWORD requested = count ? *count : 0;
        HRESULT hr = real_->GetDeviceData(cbObjectData, data, count, flags);
        if (count && data && SUCCEEDED(hr)) {
            if (*count < requested) {
                appendInjectedData(
                    kind_, cbObjectData, data, requested, count, flags
                );
            }
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE SetDataFormat(LPCDIDATAFORMAT format) override { return real_->SetDataFormat(format); }
    HRESULT STDMETHODCALLTYPE SetEventNotification(HANDLE event) override {
        const HRESULT hr = real_->SetEventNotification(event);
        if (SUCCEEDED(hr)) {
            std::lock_guard<std::mutex> lock(g_mutex);
            rememberNotificationLocked(kind_, event);
        }
        appendLog(
            "SetEventNotification W kind="
            + std::to_string(static_cast<int>(kind_))
            + " active=" + std::to_string(event != nullptr)
        );
        return hr;
    }
    HRESULT STDMETHODCALLTYPE SetCooperativeLevel(HWND hwnd, DWORD flags) override { return real_->SetCooperativeLevel(hwnd, flags); }
    HRESULT STDMETHODCALLTYPE GetObjectInfo(LPDIDEVICEOBJECTINSTANCEW out, DWORD object, DWORD how) override { return real_->GetObjectInfo(out, object, how); }
    HRESULT STDMETHODCALLTYPE GetDeviceInfo(LPDIDEVICEINSTANCEW out) override { return real_->GetDeviceInfo(out); }
    HRESULT STDMETHODCALLTYPE RunControlPanel(HWND owner, DWORD flags) override { return real_->RunControlPanel(owner, flags); }
    HRESULT STDMETHODCALLTYPE Initialize(HINSTANCE inst, DWORD version, REFGUID guid) override { return real_->Initialize(inst, version, guid); }
    HRESULT STDMETHODCALLTYPE CreateEffect(REFGUID guid, LPCDIEFFECT effect, LPDIRECTINPUTEFFECT *out, LPUNKNOWN outer) override { return real_->CreateEffect(guid, effect, out, outer); }
    HRESULT STDMETHODCALLTYPE EnumEffects(LPDIENUMEFFECTSCALLBACKW cb, LPVOID ref, DWORD type) override { return real_->EnumEffects(cb, ref, type); }
    HRESULT STDMETHODCALLTYPE GetEffectInfo(LPDIEFFECTINFOW out, REFGUID guid) override { return real_->GetEffectInfo(out, guid); }
    HRESULT STDMETHODCALLTYPE SendForceFeedbackCommand(DWORD flags) override { return real_->SendForceFeedbackCommand(flags); }
    HRESULT STDMETHODCALLTYPE GetForceFeedbackState(LPDWORD out) override { return real_->GetForceFeedbackState(out); }
    HRESULT STDMETHODCALLTYPE EnumCreatedEffectObjects(LPDIENUMCREATEDEFFECTOBJECTSCALLBACK cb, LPVOID ref, DWORD flags) override { return real_->EnumCreatedEffectObjects(cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE Escape(LPDIEFFESCAPE escape) override { return real_->Escape(escape); }
    HRESULT STDMETHODCALLTYPE Poll() override { return real_->Poll(); }
    HRESULT STDMETHODCALLTYPE SendDeviceData(DWORD cbObjectData, LPCDIDEVICEOBJECTDATA data, LPDWORD count, DWORD flags) override { return real_->SendDeviceData(cbObjectData, data, count, flags); }
    HRESULT STDMETHODCALLTYPE EnumEffectsInFile(LPCWSTR file, LPDIENUMEFFECTSINFILECALLBACK cb, LPVOID ref, DWORD flags) override { return real_->EnumEffectsInFile(file, cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE WriteEffectToFile(LPCWSTR file, DWORD count, LPDIFILEEFFECT effects, DWORD flags) override { return real_->WriteEffectToFile(file, count, effects, flags); }
    HRESULT STDMETHODCALLTYPE BuildActionMap(LPDIACTIONFORMATW format, LPCWSTR user, DWORD flags) override { return real_->BuildActionMap(format, user, flags); }
    HRESULT STDMETHODCALLTYPE SetActionMap(LPDIACTIONFORMATW format, LPCWSTR user, DWORD flags) override { return real_->SetActionMap(format, user, flags); }
    HRESULT STDMETHODCALLTYPE GetImageInfo(LPDIDEVICEIMAGEINFOHEADERW header) override { return real_->GetImageInfo(header); }

private:
    IDirectInputDevice8W *real_ = nullptr;
    DeviceKind kind_ = DeviceKind::Unknown;
    volatile LONG refs_ = 1;
};

class ProxyDirectInput8A final : public IDirectInput8A {
public:
    explicit ProxyDirectInput8A(IDirectInput8A *real) : real_(real) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, LPVOID *out) override {
        if (!out) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown)
            || IsEqualIID(riid, IID_IDirectInputA)
            || IsEqualIID(riid, IID_IDirectInput2A)
            || IsEqualIID(riid, IID_IDirectInput7A)
            || IsEqualIID(riid, IID_IDirectInput8A)) {
            *out = static_cast<IDirectInput8A *>(this);
            AddRef();
            appendLog("kept wrapped DirectInput A interface");
            return S_OK;
        }
        return real_->QueryInterface(riid, out);
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&refs_); }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG refs = InterlockedDecrement(&refs_);
        if (!refs) {
            real_->Release();
            delete this;
        }
        return refs;
    }
    HRESULT STDMETHODCALLTYPE CreateDevice(REFGUID guid, LPDIRECTINPUTDEVICE8A *out, LPUNKNOWN outer) override {
        HRESULT hr = real_->CreateDevice(guid, out, outer);
        if (SUCCEEDED(hr) && out && *out) {
            DeviceKind kind = DeviceKind::Unknown;
            if (IsEqualGUID(guid, kGuidSysMouse)) {
                kind = DeviceKind::Mouse;
            } else if (IsEqualGUID(guid, kGuidSysKeyboard)) {
                kind = DeviceKind::Keyboard;
            }
            if (kind != DeviceKind::Unknown) {
                appendLog(kind == DeviceKind::Mouse ? "wrapped mouse" : "wrapped keyboard");
                *out = new ProxyDeviceA(*out, kind);
            }
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE EnumDevices(DWORD type, LPDIENUMDEVICESCALLBACKA cb, LPVOID ref, DWORD flags) override { return real_->EnumDevices(type, cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE GetDeviceStatus(REFGUID guid) override { return real_->GetDeviceStatus(guid); }
    HRESULT STDMETHODCALLTYPE RunControlPanel(HWND owner, DWORD flags) override { return real_->RunControlPanel(owner, flags); }
    HRESULT STDMETHODCALLTYPE Initialize(HINSTANCE inst, DWORD version) override { return real_->Initialize(inst, version); }
    HRESULT STDMETHODCALLTYPE FindDevice(REFGUID guid, LPCSTR name, LPGUID out) override { return real_->FindDevice(guid, name, out); }
    HRESULT STDMETHODCALLTYPE EnumDevicesBySemantics(LPCSTR user, LPDIACTIONFORMATA format, LPDIENUMDEVICESBYSEMANTICSCBA cb, LPVOID ref, DWORD flags) override {
        return real_->EnumDevicesBySemantics(user, format, cb, ref, flags);
    }
    HRESULT STDMETHODCALLTYPE ConfigureDevices(LPDICONFIGUREDEVICESCALLBACK cb, LPDICONFIGUREDEVICESPARAMSA params, DWORD flags, LPVOID ref) override {
        return real_->ConfigureDevices(cb, params, flags, ref);
    }

private:
    IDirectInput8A *real_ = nullptr;
    volatile LONG refs_ = 1;
};

class ProxyDirectInput8W final : public IDirectInput8W {
public:
    explicit ProxyDirectInput8W(IDirectInput8W *real) : real_(real) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, LPVOID *out) override {
        if (!out) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown)
            || IsEqualIID(riid, IID_IDirectInputW)
            || IsEqualIID(riid, IID_IDirectInput2W)
            || IsEqualIID(riid, IID_IDirectInput7W)
            || IsEqualIID(riid, IID_IDirectInput8W)) {
            *out = static_cast<IDirectInput8W *>(this);
            AddRef();
            appendLog("kept wrapped DirectInput W interface");
            return S_OK;
        }
        return real_->QueryInterface(riid, out);
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&refs_); }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG refs = InterlockedDecrement(&refs_);
        if (!refs) {
            real_->Release();
            delete this;
        }
        return refs;
    }
    HRESULT STDMETHODCALLTYPE CreateDevice(REFGUID guid, LPDIRECTINPUTDEVICE8W *out, LPUNKNOWN outer) override {
        HRESULT hr = real_->CreateDevice(guid, out, outer);
        if (SUCCEEDED(hr) && out && *out) {
            DeviceKind kind = DeviceKind::Unknown;
            if (IsEqualGUID(guid, kGuidSysMouse)) {
                kind = DeviceKind::Mouse;
            } else if (IsEqualGUID(guid, kGuidSysKeyboard)) {
                kind = DeviceKind::Keyboard;
            }
            if (kind != DeviceKind::Unknown) {
                appendLog(kind == DeviceKind::Mouse ? "wrapped mouse W" : "wrapped keyboard W");
                *out = new ProxyDeviceW(*out, kind);
            }
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE EnumDevices(DWORD type, LPDIENUMDEVICESCALLBACKW cb, LPVOID ref, DWORD flags) override { return real_->EnumDevices(type, cb, ref, flags); }
    HRESULT STDMETHODCALLTYPE GetDeviceStatus(REFGUID guid) override { return real_->GetDeviceStatus(guid); }
    HRESULT STDMETHODCALLTYPE RunControlPanel(HWND owner, DWORD flags) override { return real_->RunControlPanel(owner, flags); }
    HRESULT STDMETHODCALLTYPE Initialize(HINSTANCE inst, DWORD version) override { return real_->Initialize(inst, version); }
    HRESULT STDMETHODCALLTYPE FindDevice(REFGUID guid, LPCWSTR name, LPGUID out) override { return real_->FindDevice(guid, name, out); }
    HRESULT STDMETHODCALLTYPE EnumDevicesBySemantics(LPCWSTR user, LPDIACTIONFORMATW format, LPDIENUMDEVICESBYSEMANTICSCBW cb, LPVOID ref, DWORD flags) override {
        return real_->EnumDevicesBySemantics(user, format, cb, ref, flags);
    }
    HRESULT STDMETHODCALLTYPE ConfigureDevices(LPDICONFIGUREDEVICESCALLBACK cb, LPDICONFIGUREDEVICESPARAMSW params, DWORD flags, LPVOID ref) override {
        return real_->ConfigureDevices(cb, params, flags, ref);
    }

private:
    IDirectInput8W *real_ = nullptr;
    volatile LONG refs_ = 1;
};

template <typename Fn>
Fn realProc(const char *name) {
    HMODULE module = realDInput();
    if (!module) {
        return nullptr;
    }
    return reinterpret_cast<Fn>(GetProcAddress(module, name));
}

}  // namespace

extern "C" HRESULT WINAPI DirectInput8Create(HINSTANCE inst, DWORD version, REFIID riid, LPVOID *out, LPUNKNOWN outer) {
    ensureProxyState();
    auto fn = realProc<DirectInput8CreateFn>("DirectInput8Create");
    if (!fn) {
        return E_FAIL;
    }
    appendLog("DirectInput8Create called");
    ensureCommandWatcher();
    HRESULT hr = fn(inst, version, riid, out, outer);
    if (SUCCEEDED(hr) && out && *out && IsEqualIID(riid, IID_IDirectInput8A)) {
        appendLog("wrapped IDirectInput8A");
        *out = new ProxyDirectInput8A(static_cast<IDirectInput8A *>(*out));
    } else if (SUCCEEDED(hr) && out && *out && IsEqualIID(riid, IID_IDirectInput8W)) {
        appendLog("wrapped IDirectInput8W");
        *out = new ProxyDirectInput8W(static_cast<IDirectInput8W *>(*out));
    }
    return hr;
}

extern "C" HRESULT WINAPI DllCanUnloadNow() {
    auto fn = realProc<DllCanUnloadNowFn>("DllCanUnloadNow");
    return fn ? fn() : S_FALSE;
}

extern "C" HRESULT WINAPI DllGetClassObject(REFCLSID clsid, REFIID riid, LPVOID *out) {
    auto fn = realProc<DllGetClassObjectFn>("DllGetClassObject");
    return fn ? fn(clsid, riid, out) : CLASS_E_CLASSNOTAVAILABLE;
}

extern "C" HRESULT WINAPI DllRegisterServer() {
    auto fn = realProc<DllRegisterServerFn>("DllRegisterServer");
    return fn ? fn() : E_FAIL;
}

extern "C" HRESULT WINAPI DllUnregisterServer() {
    auto fn = realProc<DllUnregisterServerFn>("DllUnregisterServer");
    return fn ? fn() : E_FAIL;
}

extern "C" LPCDIDATAFORMAT WINAPI GetdfDIJoystick() {
    auto fn = realProc<GetdfDIJoystickFn>("GetdfDIJoystick");
    return fn ? fn() : nullptr;
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_proxyModule = module;
    } else if (reason == DLL_PROCESS_DETACH && g_commandStopEvent) {
        SetEvent(g_commandStopEvent);
    }
    return TRUE;
}
