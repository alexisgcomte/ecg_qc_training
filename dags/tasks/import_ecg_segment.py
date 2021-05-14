import pandas as pd
from pyedflib import highlevel
from pyedflib import edfreader
import argparse
import re


class EdfLoader:

    # The purpose of this class is to prepare and use parameters to read an
    # edf file, then for a channel prepare create a dataframe with a filter
    # on timestamps

    def __init__(self,
                 patient: str,
                 record: str,
                 segment: str):

        default_path = '/home/DATA/lateppe/Recherche_ECG/'
        self.patient = patient
        self.record = record
        self.segment = segment
        self.edf_file = default_path + self.patient + '/EEG_' + \
            self.record + '_' + self.segment + '.edf'

        self.headers = highlevel.read_edf_header(self.edf_file)
        self.channels = self.headers['channels']

        file_pattern = self.edf_file[re.search('PAT_*',
                                               self.edf_file).start():]
        self.patient_name = file_pattern[:re.search('/', file_pattern).start()]

        self.startdate = pd.to_datetime(
            self.headers['startdate']) + pd.Timedelta(hours=1)

    def convert_edf_to_dataframe(self,
                                 channel_name: str,
                                 start_time: pd.Timestamp,
                                 end_time: pd.Timestamp) -> pd.DataFrame:

        # From its path, load an edf file for a selected channel and
        # adapt it in DataFrame, filtered by start_time and end_time

        self.fs = self.headers[
            'SignalHeaders'][
            self.channels.index(channel_name)]['sample_rate']

        with edfreader.EdfReader(self.edf_file) as f:
            signals = f.readSignal(self.channels.index(channel_name))

        freq_ns = int(1/self.fs*1_000_000_000)
        df = pd.DataFrame(signals,
                          columns=['signal'],
                          index=pd.date_range(self.startdate,
                                              periods=len(signals),
                                              freq=f'{freq_ns}ns'
                                              ))

        df = df[(df.index >= start_time) & (df.index < end_time)]

        return df


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

    args = parser.parse_args()

    loader = EdfLoader(patient=args.patient,
                       record=args.record,
                       segment=args.segment)

    df = loader.convert_edf_to_dataframe(
        channel_name=args.channel,
        start_time=pd.Timestamp(args.start_time),
        end_time=pd.Timestamp(args.end_time))

    df.to_csv(args.output_folder +
              f'/ecg_segment_{args.patient}_{args.record}_{args.channel}.csv',
              index=False)
