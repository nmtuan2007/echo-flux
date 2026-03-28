import pyaudio

pa = pyaudio.PyAudio()
default_host_api = pa.get_default_host_api_info()["index"]
print(f"Default host API: {default_host_api}")

for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info.get("maxInputChannels", 0) > 0:
        is_default_api = (info.get("hostApi") == default_host_api)
        print(f"[{i}] {info.get('name')} (HostAPI={info.get('hostApi')}, DefaultAPI={is_default_api})")

pa.terminate()
