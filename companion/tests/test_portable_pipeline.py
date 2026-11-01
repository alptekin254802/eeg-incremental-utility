"""Safety and integrity contracts for the new orchestration layer.

These tests do not fit scientific models. Run with Python -B to keep the
versioned source tree free of bytecode artifacts.
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import portable_pipeline as p


class PortableContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base/'source'
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        (self.source/'data').mkdir()
        item=self.source/'data/value.txt';item.write_text('frozen input\n',encoding='utf-8')
        mapping={'data/value.txt':p.digest(item)}
        p.write_json(self.source/'FILE_MANIFEST.json',mapping)
        return mapping

    def test_both_path_separators(self):
        self.assertEqual(p.relative_name(r'data\folder\file.csv'),'data/folder/file.csv')
        self.assertEqual(p.relative_name('data/folder/file.csv'),'data/folder/file.csv')

    def test_path_escape_and_drive_rejected(self):
        for name in ['../x','a/../x',r'a\..\x','/absolute',r'C:\absolute',r'C:relative',r'\\host\share\x','a//b','a/./b','']:
            with self.subTest(name=name),self.assertRaises(p.IntegrityError):p.relative_name(name)

    def test_duplicate_normalized_paths_rejected(self):
        h='0'*64
        with self.assertRaises(p.IntegrityError):p.normalized_manifest({'a/b':h,r'a\b':h})

    def test_missing_changed_and_unmanifested_inputs_rejected(self):
        mapping=self.fixture();self.assertEqual(p.verify_manifest(self.source,mapping,exact=True),1)
        f=self.source/'data/value.txt';f.write_text('changed',encoding='utf-8')
        with self.assertRaises(p.IntegrityError):p.verify_manifest(self.source,mapping,exact=True)
        f.write_text('frozen input\n',encoding='utf-8');extra=self.source/'unlisted.txt';extra.write_text('extra')
        with self.assertRaises(p.IntegrityError):p.verify_manifest(self.source,mapping,exact=True)
        # A missing manifest target is different from an unlisted-file error.
        with self.assertRaises(p.IntegrityError):p.verify_manifest(self.source,{'missing.txt':'0'*64})

    def test_code_changes_change_computation_fingerprint_but_docs_do_not(self):
        a={'reproduce.py':'a'*64,'tools/portable_pipeline.py':'b'*64,'docs/README.md':'c'*64}
        b={**a,'docs/README.md':'d'*64};self.assertEqual(p.computation_fingerprint(a),p.computation_fingerprint(b))
        b={**a,'tools/portable_pipeline.py':'d'*64};self.assertNotEqual(p.computation_fingerprint(a),p.computation_fingerprint(b))

    def test_output_overlap_both_directions(self):
        for target in [self.source,self.source/'nested',self.base]:
            with self.assertRaises(p.IntegrityError):p.validate_output(self.source,target)
        self.assertEqual(p.validate_output(self.source,self.base/'new'),(self.base/'new').resolve())

    def test_dataset_overlap_both_directions(self):
        dataset=self.base/'raw';dataset.mkdir()
        for target in [dataset,dataset/'run',self.base]:
            with self.assertRaises(p.IntegrityError):p.validate_output(self.source,target,dataset)

    def test_existing_checkpoint_or_configuration_rejected(self):
        out=self.base/'out';out.mkdir()
        self.assertEqual(p.validate_output(self.source,out),out.resolve())
        (out/'CONFIG.json').write_text('{"old_configuration":true}')
        with self.assertRaises(p.IntegrityError):p.validate_output(self.source,out)

    def test_symlink_escape_rejected_when_available(self):
        outside=self.base/'outside.txt';outside.write_text('outside')
        link=self.source/'link.txt'
        try:link.symlink_to(outside)
        except OSError:self.skipTest('Creating symlinks is unavailable for this account')
        with self.assertRaises(p.IntegrityError):p.verify_manifest(self.source,{'link.txt':p.digest(outside)})

    def test_staging_normalizes_only_copied_metadata(self):
        names=['extensions/utility/audit/INPUT_MANIFEST.json','extensions/device_transfer/analysis/results/INPUT_MANIFEST.json']
        for name in names:p.write_json(self.source/name,{'files':{r'a\b.csv':'0'*64},'release_root':'original/location'})
        original={name:(self.source/name).read_bytes() for name in names}
        code=self.source/'reference_release/code.py';code.parent.mkdir();code.write_text('VALUE = 7\n')
        before=p.directory_inventory(self.source)
        out=self.base/'out';out.mkdir();workspace=p.stage_workspace(self.source,out)
        for name in names:
            self.assertEqual((self.source/name).read_bytes(),original[name])
            self.assertEqual(list(p.read_json(workspace/name)['files']),['a/b.csv'])
        self.assertEqual((workspace/'reference_release/code.py').read_bytes(),code.read_bytes())
        self.assertEqual(p.directory_inventory(self.source),before)

    def test_current_figure_identity_map(self):
        self.assertEqual(len(p.FIGURES),12)
        self.assertEqual(p.FIGURES['Figure S7'],'eegnet_comparison.pdf')
        self.assertEqual(p.FIGURES['Figure 4'],'device_transfer.pdf')
        self.assertEqual(p.FIGURES['Figure 5'],'independent_adult_increment.pdf')
        self.assertEqual(p.FIGURES['Figure S4'],'model_family_sensitivity.pdf')
        self.assertEqual(p.FIGURES['Figure S6'],'controlled_simulations.pdf')

    def test_output_only_current_section_cross_reference(self):
        assets=self.base/'assets';(assets/'tables').mkdir(parents=True)
        target=assets/'tables/all_comparisons.tex'
        original='Numeric cell: 0.0267. See package (Section~S9).\n'
        target.write_text(original,encoding='utf-8')
        changes=p.current_table_crossrefs(assets)
        self.assertEqual(target.read_text(),r'Numeric cell: 0.0267. See package (Section~\ref{sec:s-data}).'+'\n')
        self.assertEqual(len(changes),1)
        with self.assertRaises(p.IntegrityError):p.current_table_crossrefs(assets)


if __name__=='__main__':unittest.main(verbosity=2)
