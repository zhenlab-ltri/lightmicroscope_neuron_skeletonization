import os
folder = '/home/zhenlab/shuyu/PyQt_related/LMS_Pipeline/IMAGES/RIH2/03252022' #''/media/zhenlab/My Passport/Images/RIH3/04072022'

file_names = sorted(os.listdir(folder))
for i in file_names:
    print(i[:-4])