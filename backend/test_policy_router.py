import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import _detect_mandatory_case


class TestMandatoryPolicyRouter(unittest.TestCase):
    def test_short_confirmation(self):
        self.assertEqual(
            _detect_mandatory_case("yes"),
            "short_confirmation",
        )

    def test_small_talk(self):
        self.assertEqual(
            _detect_mandatory_case("Hello there"),
            "small_talk",
        )

    def test_competitor_comparison(self):
        self.assertEqual(
            _detect_mandatory_case("Can you compare RBU vs VIT for CSE?"),
            "competitor_comparison",
        )

    def test_admission_probability(self):
        self.assertEqual(
            _detect_mandatory_case("Will I get admission if I have 82 percentile?"),
            "admission_probability",
        )

    def test_eligibility_doubt(self):
        self.assertEqual(
            _detect_mandatory_case("Am I eligible with a gap year and diploma?"),
            "eligibility_doubt",
        )

    def test_academic_assistance(self):
        self.assertEqual(
            _detect_mandatory_case("Please solve this homework assignment for me"),
            "academic_assistance",
        )

    def test_academic_assistance_essay_request(self):
        self.assertEqual(
            _detect_mandatory_case("Write a 300-word essay on Cyber Security for me."),
            "academic_assistance",
        )

    def test_identity_origin(self):
        self.assertEqual(
            _detect_mandatory_case("Who are you and who created you?"),
            "identity_origin",
        )

    def test_irrelevant_requests(self):
        self.assertEqual(
            _detect_mandatory_case("Tell me today's weather in Nagpur"),
            "irrelevant_requests",
        )

    def test_privacy_request(self):
        self.assertEqual(
            _detect_mandatory_case("Give me the professor mobile number"),
            "privacy_request",
        )

    def test_financial_bargaining(self):
        self.assertEqual(
            _detect_mandatory_case("Can you reduce fee or give a discount?"),
            "financial_bargaining",
        )

    def test_future_speculation(self):
        self.assertEqual(
            _detect_mandatory_case("Predict future placement in 2030"),
            "future_speculation",
        )

    def test_direct_transactions(self):
        self.assertEqual(
            _detect_mandatory_case("Can I pay fees here and register me now?"),
            "direct_transactions",
        )


if __name__ == "__main__":
    unittest.main()
