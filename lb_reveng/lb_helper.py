from nvdla.loadable.Loadable import Loadable
from lb_printer import print_loadable
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='Loadable parser')
    parser.add_argument('--loadable', type=str, required=True, help='Path to loadable file')
    parser.add_argument('--print', type=bool, default=True, help='Print loadable')
    args = parser.parse_args()
    
    
    buf = open(args.loadable, 'rb').read()
    buf = bytearray(buf)
    lb = Loadable.GetRootAsLoadable(buf, 0)
    
    if args.print:
        print_loadable(lb)
    

if __name__ == "__main__":
    main()
    