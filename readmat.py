from scipy.io import loadmat
import os
import glob

label_root = './SumMe/GT'
data_root = './frames'

with open(data_root + '/annotation.csv', 'w') as f:
    # write header
    header = 'video_name,start_frame,end_frame,gt_score\n'
    f.write(header)
    for filename in os.listdir(label_root):
        
        # standardize file naming
        new_file_name = filename.replace(' ', '_')
        if not filename == new_file_name:
            os.rename(label_root + '/' + filename, label_root + '/' + new_file_name)

        # read label from .mat
        print("Reading", new_file_name)
        data = loadmat(label_root + '/' + new_file_name)
        num_frames = data['nFrames'][0][0]
        video_name = new_file_name.split('.')[0]
        # print(' number of frames = ', len(data['gt_score']))

        # validate number of frames in dataset matches with number of labels
        if len(glob.glob(data_root + '/' + video_name + '/*.jpg')) == num_frames:
            for i in range(num_frames):
                label = data['gt_score'][i][0]
                line = f'{video_name}, {i}, {i + 1}, {label}\n'
                f.write(line)
    


    
    
