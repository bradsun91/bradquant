import pandas as pd
from pathlib import Path
import gmbp_quant.common.unittest as ut
from gmbp_quant.dpl.sec_edgar.sec_data_loader import SecEdgarDataLoader


class TestSecEdgarDataLoader(ut.TestCase):
    def test_parse_forms_to_tsv(self):
        data_dir = Path(__file__).parent / 'sec_data'
        form_type = '13F-HR'
        form_folder = '13F-HR'
        year, quarter = '2019', 'Q4'
        document_id = '0000902219-19-000479'
        sec_loader = SecEdgarDataLoader(data_dir, form_type, year, quarter)
        sec_loader.parse_forms_to_tsv()
        parsed_data_path = (data_dir / 'tsv_files' / form_folder / year /
                                quarter / f'{document_id}.tsv')
        parsed_data = pd.read_csv(parsed_data_path, sep='\t')
        benchmark_path = Path(__file__).parent / 'benchmark' / f'{document_id}.tsv'
        benchmark = pd.read_csv(benchmark_path, sep='\t')
        self.assertEqual(benchmark, parsed_data)


if __name__ == '__main__':
    ut.main()
