from checklist.test_types import MFT, INV, DIR
TYPE_MAP = {
            MFT: 'MFT',
            INV: 'INV',
            DIR: 'DIR',
        }


SUITE_PATH = {
    'sentiment' : 'sentiment_suite.pkl', 
    'qqp' : 'qqp_suite.pkl', 
    'mc' : 'squad_suite.pkl'
}

LIST_OF_TFs = ['change neutral words with BERT', 'change names', 'change numbers', 'protected: race', 'protected: sexual', 'protected: religion', \
                'protected: nationality', 'add random urls and handles', 'typos 1', 'typos 2', 'contractions', 'add positive phrases', 'add negative phrases']

