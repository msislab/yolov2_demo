import os, argparse
from pathlib import Path
from progressbar import ProgressBar
# from tqdm import tqdm

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT-path', type=str, default='', help='')
    parser.add_argument('--PD-path', type=str, default='', help='')
    # parser.add_argument('--failed_GT', type=str, default='', help='')
    # parser.add_argument('--passed_GT', type=str, default='', help='')
    opt = parser.parse_args()
    return opt

def strip_conf(opt):

    PD_no_conf = opt.PD_path+"-no-conf"

    Path(PD_no_conf).mkdir(parents=True, exist_ok=True)

    label_pd = []
    for root, dirs, files in os.walk(opt.PD_path, topdown=False):
        for name in sorted(files):
            if name.endswith('txt'):
                label_pd.append(os.path.join(root,name))

    print('\n\n')
    progress_bar = ProgressBar(len(label_pd), label='Strip confidence')
    
    # for file in tqdm(label_pd,desc="Strip confidence from predictions."):
    for file in label_pd:
        progress_bar.update()
        labels = open(file, 'r').read().splitlines()
        labels_wc = open(os.path.join(PD_no_conf,os.path.basename(file).replace('jpg','txt')), 'w')
        for line in labels:
            if len(line.split(' ')) == 6:
                cls,x,y,w,h,conf = line.split(' ')
            elif len(line.split(' ')) == 7:
                cls,x,y,w,h,conf, _ = line.split(' ')                   
            labels_wc.write(('%s ' * 5  + '\n') % (cls, x,y,w,h))


            
    print("\nPath for labels with    confidence: ", opt.PD_path )
    print("Path for labels without confidence: ", PD_no_conf )
    return PD_no_conf

# def prepare_files_list(opt):
#     print(' ')
#     pGTfPD = open ( os.path.join(os.path.dirname(opt.failed_GT),"passed_GT_failed_PD.txt"), 'w')

#     failed_GT_in = open(opt.failed_GT, 'r').read().splitlines()
#     failed_GT = open ( os.path.join(os.path.dirname(opt.failed_GT),"failed_GT.txt"), 'w')
#     failed_PD = open ( os.path.join(os.path.dirname(opt.failed_GT),"failed_PD.txt"), 'w')
#     for file in tqdm(failed_GT_in,desc="Prepare files for failed_GT, failed_PD and passedGT_failedPD"):
#         failed_GT.write(file +'\n')
#         pd_file = os.path.join(opt.PD_path+"-no-conf", os.path.basename(file))
#         failed_PD.write(pd_file+'\n')
#         pGTfPD.write(pd_file+'\n')

    
#     passed_GT_in = open(opt.passed_GT, 'r').read().splitlines()
#     passed_GT = open ( os.path.join(os.path.dirname(opt.passed_GT),"passed_GT.txt"), 'w')
#     passed_PD = open ( os.path.join(os.path.dirname(opt.passed_GT),"passed_PD.txt"), 'w')
#     for file in tqdm(passed_GT_in,desc="Prepare files for passed_GT, passed_PD and passedGT_failedPD"):
#         passed_GT.write(file +'\n')
#         pGTfPD.write(file +'\n')
#         pd_file = os.path.join(opt.PD_path+"-no-conf", os.path.basename(file))
#         passed_PD.write(os.path.join(opt.PD_path+"-no-conf", os.path.basename(file))+'\n')

#     print(f"\nTotal  images: {len(passed_GT_in)+len(failed_GT_in)}")
#     print(f"Failed images: {len(failed_GT_in)}")
#     print(f"Passed images: {len(passed_GT_in)}\n")
    
#     print("Path for failed PD: ", os.path.join(os.path.dirname(opt.failed_GT),"failed_PD.txt") )
#     print("Path for passed GT: ", os.path.join(os.path.dirname(opt.passed_GT),"passed_GT.txt\n") )
#     print("Path for passed-GT failed-PD: ", os.path.join(os.path.dirname(opt.failed_GT),"passed_GT_failed_PD.txt") )


def main(opt):
    strip_conf(opt)
    # prepare_files_list(opt)

    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)








