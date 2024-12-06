import unittest

# Import the functions from the script
# Assuming the script is named 'error_insertion.py' and functions are accessible
# If the script is in another file, adjust the import accordingly
from insert_errors import insert_errors

class TestInsertErrors(unittest.TestCase):

    def test_adjective_inflection_error(self):
        # Test inflectional errors in adjectives
        # Sentence: वह अच्छी लड़की है।
        # Expected Error: वह अच्छे लड़की है। or वह अच्छा लड़की है।
        sentence = [
            ('वह', 'PRP', []),
            ('अच्छी', 'JJ', []),
            ('लड़की', 'NN', []),
            ('है', 'VAUX', [])
        ]
        expected_err = 'वह अच्छे लड़की हैं'
        expected_err2 = 'वह अच्छा लड़की है।'
        expected_err3 = 'वह अच्छा लड़की हैं'
        expected_cor = 'वह अच्छी लड़की है'
        err, cor = insert_errors(sentence)
        self.assertIn(err, [expected_err, expected_err2, expected_err3], 'Test 1 - Error Failing')
        self.assertEqual(cor, expected_cor, 'Test 1 - Correct Failing')

    def test_verb_inflection_error(self):
        # Test inflectional errors in verbs
        # Sentence: वे स्कूल जाते हैं।
        # Expected Error: वे स्कूल जाता हैं। or वे स्कूल जाती है। or वे स्कूल जाता है।
        sentence = [
            ('वे', 'PRP', []),
            ('स्कूल', 'NN', []),
            ('जाते', 'VM', []),
            ('हैं', 'VAUX', [])
        ]
        expected_err = 'वे स्कूल जाती है'
        expected_err2 = 'वे स्कूल जाता हैं'
        expected_err3 = 'वे स्कूल जाता है'
        expected_cor = 'वे स्कूल जाते हैं'
        err, cor = insert_errors(sentence)
        self.assertIn(err, [expected_err, expected_err2, expected_err3], 'Test 2 - Error Failing')
        self.assertEqual(cor, expected_cor, 'Test 2 - Correct Failing')

    def test_pronoun_inflection_error(self):
        # Test inflectional errors in pronouns
        # Sentence: उसने अपना काम पूरा किया।
        # Expected Error: उसने अपनी काम पूरा किया। or उसने अपनी काम पूरा कि। or उसने अपनी काम पूरा किए।
        sentence = [
            ('उसने', 'PRP', []),
            ('अपना', 'PRP', []),
            ('काम', 'NN', []),
            ('पूरा', 'JJ', []),
            ('किया', 'VM', [])
        ]
        expected_err = 'उसने अपनी काम पूरा किया'
        expected_err2 = 'उसने अपनी काम पूरा कि' 
        expected_err3 = 'उसने अपनी काम पूरा किए' 

        expected_cor = 'उसने अपना काम पूरा किया'
        err, cor = insert_errors(sentence)
        self.assertIn(err, [expected_err, expected_err2, expected_err3], 'Test 3 - Error Failing')
        self.assertEqual(cor, expected_cor, 'Test 3 - Correct Failing')

    def test_postposition_inflection_error(self):
        # Test inflectional errors in postpositions
        # Sentence: राम के पास किताब है।
        # Expected Error: राम का पास किताब है।
        sentence = [
            ('राम', 'NNP', []),
            ('के', 'PSP', []),
            ('पास', 'NN', []),
            ('किताब', 'NN', []),
            ('है', 'VAUX', [])
        ]
        expected_err = 'राम का पास किताब है'
        expected_err2 = 'राम का पास किताब हैं'
        expected_err3 = 'राम की पास किताब हैं'
        expected_cor = 'राम के पास किताब है'
        err, cor = insert_errors(sentence)
        self.assertIn(err, [expected_err, expected_err2, expected_err3], 'Test 4 - Error Failing')
        self.assertEqual(cor, expected_cor, 'Test 4 - Correct Failing')

    def test_exception_handling(self):
        # Test handling of words in the exceptions list
        # Sentence: वह खुश है।
        # Expected: No change since 'है' is in exceptions
        sentence = [
            ('वह', 'PRP', []),
            ('खुश', 'JJ', []),
            ('है', 'VAUX', [])
        ]
        expected_err = 'वह खुश है'
        expected_err2 = 'वह खुश हैं'
        expected_cor = 'वह खुश है'
        err, cor = insert_errors(sentence)
        self.assertIn(err, [expected_err, expected_err2], 'Test 5 - Error Failing')
        self.assertEqual(cor, expected_cor, 'Test 5 - Correct Failing')

if __name__ == '__main__':
    unittest.main()
