from eval_od import eval_od
import pandas as pd
import os

def grid_search(args):
    output_xlsx = os.path.join(args.pred_dir, f'-{args.confidence_mode}-{args.tp_mode}.xlsx')
    columns = ['bin_thresh', 'pos_thresh', 'tp_thresh', 'tp', 'fp', 'fn', 'tn', 'sensitivity', 'precision']
    df = pd.DataFrame(columns=columns)
    count = 0
    for tp_t in args.tp_threshs:
        for bin_t in args.bin_threshs:
            for pos_t in args.pos_threshs:
                _, _, metric_dict = eval_od(
                    args.root_dir, args.pred_dir,
                    bin_t, 
                    args.confidence_mode, pos_t,
                    args.tp_mode, tp_t)
                tp, fp, fn, tn = metric_dict['total']
                sensi = tp / (tp + fn)
                df.loc[count] = [bin_t, pos_t, tp_t, tp, fp, fn, tn, sensi, 1-fp/(tp+fp)]
                count += 1
    df = df.sort_values(by=['sensitivity', 'precision'], ascending=False)
    print(df)
    df.to_excel(output_xlsx)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', action='store', type=str, required=True)
    parser.add_argument('--pred_dir', action='store', type=str, required=True)
    parser.add_argument('--bin_threshs', action='store', nargs='+', type=float, required=True)
    parser.add_argument('--pos_threshs', action='store', nargs='+', type=float, required=True)
    parser.add_argument('--confidence_mode', action='store', type=str, required=True)
    parser.add_argument('--tp_mode', action='store', type=str, required=True)
    parser.add_argument('--tp_threshs', action='store', nargs='+', type=float, required=True)

    args = parser.parse_args()
    grid_search(args)