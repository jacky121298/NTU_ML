import os
import sys
import torch
import pickle
import numpy as np

def encode16(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))

def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))

if __name__ == '__main__':
    pruned_net = sys.argv[1]
    model_dir = sys.argv[2]

    print(f"original cost: {os.stat(pruned_net).st_size} bytes.")
    params = torch.load(pruned_net)

    encode16(params, f'{model_dir}/16_bit_model.pkl')
    print(f"16-bit cost: {os.stat(f'{model_dir}/16_bit_model.pkl').st_size} bytes.")

    encode8(params, f'{model_dir}/8_bit_model.pkl')
    print(f"8-bit cost: {os.stat(f'{model_dir}/8_bit_model.pkl').st_size} bytes.")