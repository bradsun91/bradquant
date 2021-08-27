import sys
import argparse

from sec_data_loader import SecEdgarDataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./sec_data', help='SEC form data folder')
parser.add_argument('--form', type=str, help='SEC form type')
parser.add_argument('--year', type=str, help='SEC data year')
parser.add_argument('--quarter', type=str, help='SEC data quarter')
opt = parser.parse_args()


if __name__ == '__main__':
    years = opt.year.split(',')
    quarters = opt.quarter.split(',')
    for year in years:
        for quarter in quarters:
            sec = SecEdgarDataLoader(opt.data, opt.form, year, quarter)
            if len(sec.form_list) != 0:
                print('Parsing SEC data to tsv files ...')
                sec.parse_forms_to_tsv()
                print('Inserting parsed SEC data to DB ...')
                sec.insert_db()