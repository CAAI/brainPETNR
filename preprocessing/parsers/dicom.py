from pydicom import dcmread

class DicomParser:
    def __init__(self, dcm_file):
        self.dcm_file = dcm_file
        self.info = {}
    
    def get_header_info(self):
        header = dcmread(self.dcm_file, stop_before_pixels=True)
        for h in header:
            # deal with embedded data
            if 'Sequence' in h.keyword:
                for subh in h:
                    for v in subh:
                        if 'Sequence' in v.keyword:
                            for subv in v:
                                for subsubv in subv:
                                    val = self.format_val(subsubv.value)
                                    self.info[subsubv.keyword] = val
                        else:
                            val = self.format_val(v.value)
                            self.info[v.keyword] = val
            else:
                val = self.format_val(h.value)
                self.info[h.keyword] = val
                
    
    @staticmethod
    def format_val(x):
        if type(x) == int:
            xf = x
        elif type(x) == bytes:
            try:
                xf = x.decode('utf-16')
            except Exception as e:
                xf = '???'
        else:
            xf = str(x)
        return xf