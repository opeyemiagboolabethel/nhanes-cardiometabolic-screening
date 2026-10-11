import json,hashlib,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.pipeline import ROOT,FEATURES,cohort,metrics,predict

class EvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=json.loads((ROOT/'outputs/summary.json').read_text());cls.b=joblib.load(ROOT/'outputs/screening_model.joblib');cls.dev,_=cohort(lambda k:k+'_I','WTMEC2YR');cls.test,_=cohort(lambda k:'P_'+k,'WTMECPRP')
    def test_source_checksums(self):
        for item in json.loads((ROOT/'data/source_manifest.json').read_text()):self.assertEqual(hashlib.sha256((ROOT/'data/raw'/item['file']).read_bytes()).hexdigest(),item['sha256'])
    def test_no_release_or_psu_leakage(self):
        self.assertTrue(set(self.dev.SEQN).isdisjoint(self.test.SEQN))
        split=pd.read_csv(ROOT/'outputs/development_partitions.csv');self.assertEqual(split.groupby('group').partition.nunique().max(),1)
    def test_outcome_excluded_from_inputs(self):
        self.assertEqual(FEATURES,['RIDAGEYR','RIAGENDR','BMXBMI','BMXWAIST']);self.assertNotIn('LBXGH',self.b['model'].feature_names_in_)
        self.assertTrue((self.dev.DIQ010!=1).all());self.assertTrue((self.test.DIQ050!=1).all());self.assertTrue((self.test.DIQ070!=1).all())
    def test_saved_predictions_reproduce_metrics(self):
        p=pd.read_csv(ROOT/'outputs/temporal_predictions.csv');m=metrics(p.target.to_numpy(),p.probability.to_numpy(),p.weight.to_numpy(),self.s['threshold'])
        for k,v in m.items():self.assertAlmostEqual(v,self.s['temporal_metrics'][k],places=10)
        np.testing.assert_allclose(p.probability,predict(self.b,self.test[FEATURES]),rtol=1e-12)
    def test_application_predictions_and_ci(self):
        X=pd.DataFrame([[20,2,20,70],[80,1,35,120]],columns=FEATURES);p=predict(self.b,X)
        self.assertTrue(((p>0)&(p<1)).all());self.assertTrue(p[1]>p[0]);ci=self.s['conditional_test_ci']['roc_auc'];self.assertLess(ci['low'],ci['high'])
    def test_app_scenario(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run();self.assertFalse(app.exception)
        app.button[0].click().run();self.assertFalse(app.exception)
        self.assertTrue(any('Estimated elevated' in m.label for m in app.metric))

if __name__=='__main__':unittest.main()
