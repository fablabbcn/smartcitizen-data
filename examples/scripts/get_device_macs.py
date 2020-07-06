from scdata.io.read_api import ScApiDevice

print ('Getting device 13045')
device = ScApiDevice('13045')
print (device.get_device_location())
print (device.get_mac())

print ('Getting devices in Barcelona')
wm = ScApiDevice.get_world_map(city = 'Barcelona', max_date = '2020-05-01')
print ('World map get successful')

for kit in wm:
    device = ScApiDevice(kit)
    mac = device.get_mac() 
    if mac:
        print (kit, mac)