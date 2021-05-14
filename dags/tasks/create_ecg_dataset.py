from import_ecg_segment import EdfLoader
from import_annotations import make_result_df
import re
import pandas as pd
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("-p", "--patient", dest="patient",
                        help="patient to load", metavar="FILE")
    parser.add_argument("-r", "--record", dest="record",
                        help="record to load", metavar="FILE")
    parser.add_argument("-s", "--segment", dest="segment",
                        help="segment to load", metavar="FILE")
    parser.add_argument("-c", "--channel", dest="channel",
                        help="channel to load", metavar="FILE")
    parser.add_argument("-st", "--start_time", dest="start_time",
                        help="start time for filter", metavar="FILE")
    parser.add_argument("-et", "--end_time", dest="end_time",
                        help="end time for filter", metavar="FILE")
    parser.add_argument("-o", "--output_folder", dest="output_folder",
                        help="output_folder_for_ddf", metavar="FILE",
                        default="./exports")
    parser.add_argument("-ids", "--annot_ids", dest="annot_ids",
                        help="ids of annotators", metavar="FILE")

    args = parser.parse_args()
    annotators = [int(annotator) for annotator in
                  re.split(',', args.annot_ids)]

    loader = EdfLoader(patient=args.patient,
                       record=args.record,
                       segment=args.segment)

    df_ecg = loader.convert_edf_to_dataframe(
        channel_name=args.channel,
        start_time=pd.Timestamp(args.start_time),
        end_time=pd.Timestamp(args.end_time))

    df_annotation = make_result_df(ids=annotators,
                                   record=args.record,
                                   channel=args.channel,
                                   start_date=args.start_time,
                                   end_date=args.end_time)


    df = pd.concat([df_ecg, df_annotation], axis=1).dropna()

    df.to_csv(args.output_folder +
              f'/ecg_annoted_{args.patient}_{args.record}_{args.channel}.csv',
              index=False)
