DEMO_SAMPLES = [
    {
        "question": "Mia has 12 stickers. She buys 8 more and gives 5 to her friend. How many stickers does Mia have now?",
        "answer": "#### 15",
    },
    {
        "question": "A box has 6 rows of pencils with 4 pencils in each row. How many pencils are in the box?",
        "answer": "#### 24",
    },
    {
        "question": "Sam read 18 pages on Monday and twice as many pages on Tuesday. How many pages did he read in total?",
        "answer": "#### 54",
    },
    {
        "question": "A jacket costs $80 and is discounted by 25 percent. What is the sale price?",
        "answer": "#### 60",
    },
    {
        "question": "There are 45 students. One third of them join the math club. How many students join the math club?",
        "answer": "#### 15",
    },
]


class DemoDataset(list):
    def select(self, indices):
        return DemoDataset([self[i] for i in indices])


def load_demo_dataset(n_samples: int | None = None, start_idx: int = 0):
    end_idx = len(DEMO_SAMPLES) if n_samples is None else min(len(DEMO_SAMPLES), start_idx + n_samples)
    return DemoDataset(DEMO_SAMPLES[start_idx:end_idx])
