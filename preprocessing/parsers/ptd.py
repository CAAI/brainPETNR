import os
from pathlib import Path, PurePath
import re

class BackwardParser:
    def __init__(self, file_in):
        self.file_in = file_in
        
    def read_file_backward(self):
        # Open file for reading in binary mode
        with open(self.file_in, 'rb') as read_obj:
            # Move the cursor to the end of the file
            read_obj.seek(0, os.SEEK_END)
            # Get the current position of pointer i.e eof
            pointer_location = read_obj.tell()
            # Create a buffer to keep the last read line
            buffer = bytearray()
            # Loop till pointer reaches the top of the file
            while pointer_location >= 0:
                # Move the file pointer to the location pointed by pointer_location
                read_obj.seek(pointer_location)
                # Shift pointer location by -1
                pointer_location = pointer_location -1
                # read that byte / character
                new_byte = read_obj.read(1)
                new_byte
                # If the read byte is not end of line character then continue reading
                if new_byte != b'\n':
                    # If last read character is not eol then add it in buffer
                    buffer.extend(new_byte)
                else:
                    yield buffer
                    # Reinitialize the byte array to save next line
                    buffer = bytearray()
            # As file is read completely, if there is still data in buffer, then its the first line.
            if len(buffer) > 0:
                # Yield the first line too
                yield buffer
                
    def read_tail(self, stopword='DICM'):
        # read the LM file backward
        stopword = stopword[::-1]
        for byteline in self.read_file_backward():
            line = ''
            for c in byteline:
                if 31 < c < 126: # or 190 < c < 255:
                    line += chr(c)
                elif c == 255:
                    line += ''
                elif 0 < c < 32:
                    if stopword in line:
                        break
                    if len(line) > 3:
                        self.lines.append(line[::-1].strip())
                    line = ''
                
            if len(line) > 3:
                self.lines.append(line[::-1].strip())
            if stopword in line:
                break
        
class ListmodeFileParser(BackwardParser):
    def __init__(self, folder_in):
        self.folder_in = folder_in
        file_in = self.get_file()
        super().__init__(file_in)
        self.lines = []
        self.info = {}
    
    def get_file(self):
        if not isinstance(self.folder_in, PurePath):
            self.folder_in = Path(self.folder_in)
        ptd_files = [f for f in self.folder_in.iterdir() if '.ptd' in f.name]
        sorted_files = sorted(ptd_files, key = os.path.getsize, reverse=True)
        # listmode file is the largest file in the folder
        return sorted_files[0]
        
    def get_primary_info(self, include='=', exclude='!'):
        # get info from lines containing :=
        lines_with_equal = [l.replace('%', '').strip() for l in self.lines if (include in l) and (exclude not in l)]
        info = [l.split(':=') for l in lines_with_equal]
        info = [i for i in info if len(i) == 2]
        self.info = {k.strip(): v.strip() for k, v in info}
    
    def get_secondary_info(self):
        info2 = [l.strip() for l in self.lines if '=' not in l]
        # StudyInstanceUID located 1 above 'HFS'
        uid_idx = info2.index('HFS') - 1
        self.info['StudyInstanceUID'] = info2.pop(uid_idx)[3:].strip()
        # StationName (scanner type)
        station_name_idx = info2.index('PET Raw Data') + 2
        self.info['StationName'] = info2.pop(station_name_idx).strip()
        
        for i, v in enumerate(info2):
            self.info[f"add_info_{i:02}"] = v
    
    def clean_info(self):
        # change string to int if possible
        for k, v in self.info.items():
            try:
                self.info[k] = int(v)
            except Exception:
                pass

    def save_info(self, file_out):
        with open(file_out, 'w') as f:
            for item in self.lines:
                f.write(f"{item}\n")


class PetctFileParser(BackwardParser):
    def __init__(self, folder_in):
        self.folder_in = folder_in
        file_in = self.get_file()
        super().__init__(file_in)
        self.lines = []
        self.info = {}
    
    def get_file(self):
        if not isinstance(self.folder_in, PurePath):
            self.folder_in = Path(self.folder_in)
        ct_file = [f for f in self.folder_in.iterdir() if 'PETCT_SPL' in f.name]
        if ct_file:
            return ct_file[0]
        else:
            return None
    
    def get_primary_info(self):
        for l in self.lines:
            try:
                # parse info with pattern <info_name>info</info_name>
                x = re.findall("<(.*)>(.*)</", l)[0]
                self.info[x[0].strip()] = x[1].strip()
            except Exception as e:
                pass
    
    def clean_info(self):
        # change string to int if possible
        for k, v in self.info.items():
            try:
                self.info[k] = int(v)
            except Exception as e:
                pass
            

#################################################
if __name__ == '__main__':
    # data_path = '/homes/raphael/Projects/Lowdose_PiB/raw_data_ordered_anonymized'
    data_path = 'Z:\\Projects\\Lowdose_PiB\\raw_data_test_anonymized'
    fulldose_folders = [f for f in os.listdir(data_path) 
                        if 'lowdose' not in f and not os.path.isfile(os.path.join(data_path, f))]
#     fulldose_folders = ['PiB_000_000', 'PiB_001_000', 'PiB_001_001']
    output_path = os.path.join(data_path, 'Listmode_file_info')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for c, f in enumerate(fulldose_folders):
        print(f"Parsing folder {c:03} of {len(fulldose_folders):03}", end='\r')
        if 'Listmode_file_info' in f:
            continue
        p_folder = os.path.join(data_path, f)
        save_file = os.path.join(output_path, f"LM_info_{f}.txt")
        if not os.path.exists(save_file):
            parser = ListmodeFileParser(p_folder)
            parser.read_tail(stopword='DICM')
            parser.save_info(save_file)

