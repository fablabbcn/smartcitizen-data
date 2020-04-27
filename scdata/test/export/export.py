def export_to_csv(self, path = '', forced_overwrite = False):
    export_ok = True

    if path == '': epath = join(paths['dataDirectory'], 'processed', self.full_name[0:4], self.full_name[5:7], self.full_name, 'processed')
    else: epath = path

    # Export to csv
    for device in self.devices.keys():
        export_ok &= self.devices[device].export(epath, forced_overwrite = forced_overwrite)

    if export_ok: std_out(f'Test {self.full_name} exported successfully', 'SUCCESS')
    else: std_out(f'Test {self.full_name} not exported successfully', 'ERROR')

    return export_ok