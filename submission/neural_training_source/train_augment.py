"""Train/tune experiment with parent-safe paired synthetic augmentation."""
import argparse,sys,json
from pathlib import Path
from augmentation import augment_training

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--module-dir',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--synthetic-cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--synthetic-weight',type=float,default=.35)
    ap.add_argument('--prefix-features',action='store_true');args,rest=ap.parse_known_args()
    sys.path.insert(0,str(args.module_dir.resolve()))
    import common,train
    real=common.load_cache(args.cache,'train')
    augmented,protocol=augment_training(real,args.synthetic_cache,args.synthetic_weight,args.prefix_features)
    original_loader=common.load_cache
    train.load_cache=lambda path,split:augmented if split=='train' else original_loader(path,split)
    train.CONFIG_EXTRA={'augmentation':protocol}
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'augmentation_protocol.json').write_text(json.dumps(protocol,indent=2))
    print('AUGMENTED_TRAINING',json.dumps({k:v for k,v in protocol.items() if k!='source_metadata'}),flush=True)
    sys.argv=['train.py','--cache',str(args.cache),'--out',str(args.out),*rest]
    train.main()

if __name__=='__main__':main()
