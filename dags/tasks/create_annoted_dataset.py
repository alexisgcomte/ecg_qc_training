"""create_ecg_dataset script

This script wrap import_ecg_segment and import_annotation_dataset scripts
and consolidate both DataFrames.

This file can also be imported as a module and contains the following
fonctions:
    * main - the main function of the script
"""

from import_ecg_segment import EdfLoader
from import_annotations import make_annot_df
import pandas as pd
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument('-p',
                        '--patient',
                        dest='patient',
                        help='patient to load',
                        metavar='FILE')
    parser.add_argument('-r',
                        '--record',
                        dest='record',
                        help='record to load',
                        metavar='FILE')
    parser.add_argument('-ch',
                        '--channel',
                        dest='channel',
                        help='channel to load',
                        metavar='FILE')
    parser.add_argument('-sg',
                        '--segment',
                        dest='segment',
                        help='segment to load',
                        metavar='FILE')
    parser.add_argument('-st',
                        '--start_time',
                        dest='start_time',
                        help='start time for filter',
                        metavar='FILE')
    parser.add_argument('-et',
                        '--end_time',
                        dest='end_time',
                        help='end time for filter',
                        metavar='FILE')
    parser.add_argument('-ids',
                        '--annot_ids',
                        dest='annot_ids',
                        help='ids of annotators',
                        metavar='FILE')
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        help='output_folder_for_df',
                        metavar='FILE',
                        default='./exports')
    parser.add_argument('-s', '--sampling_frequency_hz',
                        dest='sampling_frequency_hz',
                        help='sampling_frequency_hz_of_file',
                        metavar='FILE',
                        default='256')
    args = parser.parse_args()

    loader = EdfLoader(patient=args.patient,
                       record=args.record,
                       segment=args.segment)

    df_ecg = loader.convert_edf_to_dataframe(
        channel_name=args.channel,
        start_time=pd.Timestamp(args.start_time),
        end_time=pd.Timestamp(args.end_time))

    df_annot = make_annot_df(ids=args.annot_ids,
                             record=args.record,
                             channel=args.channel,
                             start_date=args.start_time,
                             end_date=args.end_time,
                             sampling_frequency_hz=float(
                                 args.sampling_frequency_hz))

    df_ecg_annot = pd.concat([df_ecg, df_annot], axis=1).dropna()

    df_ecg_annot.to_csv(f'{args.output_folder}/ecg_annoted_{args.patient}'
                        f'_{args.record}_{args.channel}.csv',
                        index=False)
