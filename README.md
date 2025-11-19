# DeepUMQA
Ultrafast Shape Recognition-based Protein Model Quality Assessment using Deep Learning

# Developer:
            Saisai Guo and Jun Liu
            College of Information Engineering
            Zhejiang University of Technology, Hangzhou 310023, China
            Email: ssgmamba0824@163.com, junl@zjut.edu.cn

# Contact:
            Guijun Zhang, Prof
            College of Information Engineering
            Zhejiang University of Technology, Hangzhou 310023, China
            Email: zgj@zjut.edu.cn

# INSTALLATION
- Python > 3.5
- PyTorch 1.3
- PyRosetta
- Tested on Ubuntu 20.04 LTS

# RUNNING
```
DeepUMQA.py 

arguments:
  input                 path to input
  output                path to output (folder path, npz, or csv)

optional arguments:
  -h, --help            show this help message and exit
  --pdb, -pdb           Running on a single pdb 
  --csv, -csv           Writing results to a csv file 
  --per_res_only, -pr   Writing per-residue accuracy only 
  --leaveTempFile, -lt  Leaving temporary files 
  --process PROCESS, -p PROCESS
                       
  --featurize, -f       Running only the featurization part 
  --reprocess, -r       Reprocessing all feature files 
  --verbose, -v         Activating verbose flag 
  --ensemble, -e        Running with ensembling of 4 models. 
                   
```

1. Predicting
            
            # Running on a folder of pdbs
            
            python DeepUMQA.py -r -v input/ output/

            # Running on a single pdb file

            python DeepUMQA.py -r -v --pdb pdbfile

2. Feature extracting

python DeepUMQA.py --featurize input/ outputFea/

3. Traning

python train.py  models/

# DISCLAIMER
The executable software and the source code of DeepUMQA is distributed free of charge as it is to any non-commercial users. The authors hold no liabilities to the performance of the program.
