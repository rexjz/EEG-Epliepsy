import os
patients = ["p10"]

metadata = {
    "p10": {
        "records": [{
            "name": "p10_Record1.edf",
            "seizure_time": [(7, 36, 38, 445)]
        }, {
            "name": "p10_Record2.edf",
            "seizure_time": [(6, 29, 14, 305)]
        }],
    },
    "p11": {
        "records": [
            {
                "name": "p11_Record1.edf",
                "seizure_time":[(15, 8, 55, 64)],
            },
            {
                "name":  "p11_Record2.edf",
                "seizure_time": [(18, 11, 38, 51)]
            },
            {
                "name":  "p11_Record3.edf",
                "seizure_time": [(19, 18, 38, 91),
                   (20, 6, 38, 83),
                   (20, 53, 22, 76),
                   (21, 27, 24, 73)]
            },
            {
                "name": "p11_Record4.edf",
                "seizure_time": [(23, 55, 35, 1358)]
            }
        ]
    },
    "p12": {
        "records": [
            {
                "name": "p12_Record1.edf",
                "seizure_time": [(1, 56, 14, 76),
                   (2, 20, 12, 104)]
            },
            {
                "name": "p12_Record2.edf",
                "seizure_time": [(5, 50, 55, 118)]
            },
            {
                "name": "p12_Record3.edf",
                "seizure_time": [(6, 40, 30, 82),  # no need for the replicates
                   (7, 21, 36, 164),
                   (8, 37, 48, 113)]
            }
        ]
    },
    "p13": {
        "records": [
            {
                "name": "p13_Record1.edf",
                "seizure_time": [(2, 31, 9, 52)]
            },
            {
                "name": "p13_Record2.edf",
                "seizure_time": [(3, 33, 4, 25),
                   (4, 38, 59, 16)]
            },
            {
                "name": "p13_Record3.edf",
                "seizure_time": [(6, 45, 51, 18)]
            },
            {
                "name": "p13_Record4.edf",
                "seizure_time": [(10, 51, 41, 30),
                   (12, 18, 22, 24)]
            }
        ]
    },
    "p14": {
        "records": [
            {
                "name": "p14_Record1.edf",
                "seizure_time": [(14, 32, 2, 28),
                   (15, 34, 32, 134)]
            },
            {
                "name": "p14_Record2.edf",
                "seizure_time": [(16, 20, 58, 32),  # no need for the replicates
                   (17, 50, 56, 10)]
            },
            {
                "name": "p14_Record3.edf",
                "seizure_time": [(20, 20, 46, 31),
                   (21, 2, 4, 26),
                   (21, 27, 49, 40),
                   (21, 50, 24, 40)]
            },
        ]
    }
}