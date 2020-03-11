import torch
import argparse


def average_model_parameters(ifiles, ofile, key):
    imodels = [torch.load(ifile)[key] for ifile in ifiles]
    omodel = imodels[0]
    for imodel in imodels[1:]:
        for k, v in imodel.items():
            omodel[k] += v
    for k in imodel.keys():
        omodel[k] /= len(imodels)
    torch.save(omodel, ofile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key', default='model', type=str,
                        help='model key in the saved checkpoint (default: model).')
    parser.add_argument('ifiles', nargs='+')
    parser.add_argument('ofile')
    args = parser.parse_args()
    print(args)
    average_model_parameters(args.ifiles, args.ofile, args.key)
