import os
from joblib import Parallel, delayed
import re
import random
from absl import flags
from absl import app


# Assign flags
FLAGS = flags.FLAGS
flags.DEFINE_string("source_dir", None, "Path to slide tile files")
flags.DEFINE_string("out_dir", None, "Path for saving renamed and augmented tile files")
flags.DEFINE_string("sort_file", None, "Path to sort file / sample sheet file")
'''
    Sample sheet file must have these columns:
    column 1 = Slide name from which tile files were derived.
    column 2 = Label (e.g. CA versus Benign)
    column 3 = Group (e.g test_, train_ and valid_)
    column 4 = Number of rotation (for data augmentation, 1 for no augmentation)
    The file must be in a tab-delimited format.
'''
flags.DEFINE_integer("jobs",-1,"Number of parallel jobs (default = -1 for using all available threads)")


# Required flags
flags.mark_flag_as_required("source_dir")
flags.mark_flag_as_required("out_dir")
flags.mark_flag_as_required("sort_file")


#Read GDC sample sheet (With header)
def readGDC(filename, inpath, outpath, job, header = True):
    job = int(job)
    with open(filename, 'r') as f:
        if header:
            f.readline()
        for line in f:
            lst = line.rstrip().split('\t')
            fname = lst[0].strip()
            path = os.path.abspath(inpath)+"/"+ fname.rstrip("_files")+"_files" 
            
            CP = (cp_list(fname,inpath,outpath,lst[1],lst[2]))
            ROTATE = (rotate_list(fname,inpath,outpath,lst[1],lst[2],lst[3]))
            Parallel(n_jobs=job, verbose=1, backend="threading")(map(delayed(os.system), CP))
            Parallel(n_jobs=job, verbose=1, backend="threading")(map(delayed(os.system), ROTATE))
            
def cp_list(fname,inpath,outpath,label,group):
    lst = []
    path = os.path.abspath(inpath)+"/"+ fname.rstrip("_files")+"_files" 
    for (dirpath, dirnames, filenames) in os.walk(path):
        for ff in filenames:
            cmd = "cp " + dirpath+"/"+ff + " " + os.path.abspath(outpath)+"/"+label.strip()+"/"+group.strip()+fname+"_"+ff
            lst.append(cmd)
    return(lst)

def rotate_list(fname,inpath,outpath,label,group,nrotate):
    lst = []
    rotate_dict ={
        1:"FH_",
        2:"FV_",
        3:"R90_",
        4:"R180_",
        5:"R270_"
    }
    if int(nrotate) <= 1:
        return(lst)
    path = os.path.abspath(inpath)+"/"+ fname.rstrip("_files")+"_files" 
    for (dirpath, dirnames, filenames) in os.walk(path):
        for ff in filenames:
            for j in random.sample(rotate_dict.keys(),int(nrotate)-1):
                fl = dirpath+"/"+ff
                npath = os.path.abspath(outpath)+"/"+label.strip()+"/"+group.strip()+rotate_dict.get(j)+fname+"_"+ff
                cmd = 'singularity run --app flip DeepPATHv4.sif -i %s -o %s -s %s' % (fl,j,npath)
                lst.append(cmd)
    return(lst)
            
def main(argv):
    del argv #Unused
    readGDC(FLAGS.sort_file, FLAGS.source_dir, FLAGS.out_dir,  job = FLAGS.jobs, header = False)


if __name__ == '__main__':
    app.run(main)
