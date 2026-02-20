class TaskConfig:
    def __init__(self, task_name, labels, ignore_labels):
        self.task_name = task_name
        self.labels = labels
        self.ignore_labels = ignore_labels

    @staticmethod
    def create(task_name, labels, ignore_labels):
        return TaskConfig(task_name, labels, ignore_labels)


def get_task_config(task_name: str) -> TaskConfig:
    if task_name == "punctuation":
        return TaskConfig.create(
            task_name="punctuation",
            labels=['O', '，', '。', '：', '、', '；', '？', '！'],
            ignore_labels=['O']
        )
    else:  # segmentation
        return TaskConfig.create(
            task_name="segmentation",
            labels=['B', 'M', 'E', 'S'],
            ignore_labels=[]
        )
