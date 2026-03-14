# -*- coding: utf-8 -*-

from ast import literal_eval
from configparser import ConfigParser
from argparse import Namespace


class Config(ConfigParser):
    """
    Lớp Config mở rộng ConfigParser để đọc và phân tích các file cấu hình.
    Các giá trị trong file cấu hình sẽ được chuyển đổi tự động sang các kiểu dữ liệu
    của Python (như int, float, list, boolean, v.v.) bằng ast.literal_eval() 
    và lưu trữ vào một không gian tên (Namespace) giúp truy cập dễ dàng dưới dạng thuộc tính (ví dụ: config.batch_size).
    """

    def __init__(self, path):
        """
        Khởi tạo lớp Config bằng đường dẫn tới file cấu hình.
        
        Args:
            path (str): Đường dẫn đến file cấu hình (.ini, .cfg).
        """
        super(Config, self).__init__()

        # Đọc nội dung file cấu hình
        self.read(path)
        
        # Khởi tạo một đối tượng Namespace để lưu trữ các tham số dưới dạng thuộc tính
        self.namespace = Namespace()
        
        # Lặp qua tất cả các phần (section) và các mục (items) trong file cấu hình,
        # dùng literal_eval để chuyển giá trị chuỗi thành kiểu dữ liệu tương ứng của Python
        # dictionary comprehensions sau đó sẽ được cập nhật vào namespace thông qua hàm update()
        self.update(dict((name, literal_eval(value))
                         for section in self.sections()
                         for name, value in self.items(section)))

    def __repr__(self):
        """
        Hàm trả về chuỗi biểu diễn của đối tượng Config,
        giúp in ra các tham số cấu hình dưới dạng bảng trực quan và dễ đọc.
        """
        s = line = "-" * 15 + "-+-" + "-" * 25 + "\n"
        s += f"{'Param':15} | {'Value':^25}\n" + line
        # Trích xuất tất cả các biến đã lưu trong namespace và in theo quy định định dạng
        for name, value in vars(self.namespace).items():
            s += f"{name:15} | {str(value):^25}\n"
        s += line

        return s

    def __getattr__(self, attr):
        """
        Phương thức này tự động được gọi khi truy cập vào một thuộc tính không tồn tại trực tiếp trong đối tượng Config.
        Nó sẽ chuyển hướng truy xuất đến thuộc tính được lưu trong namespace.
        
        Args:
            attr (str): Tên thuộc tính cần truy cập.
            
        Returns:
            Giá trị của thuộc tính, hoặc None nếu nó không tồn tại.
        """
        return getattr(self.namespace, attr, None)

    def __getstate__(self):
        """
        Hỗ trợ quá trình serialize (mã hóa, ví dụ: dumper của pickle).
        Trả về toàn bộ các thuộc tính của lớp dưới dạng dictionary.
        """
        return vars(self)

    def __setstate__(self, state):
        """
        Hỗ trợ deserialize (khôi phục trạng thái, ví dụ: truyền vào từ pickle.load).
        Cập nhật lại __dict__ của class từ state dictionary được truyền vào.
        """
        self.__dict__.update(state)

    def update(self, kwargs):
        """
        Cập nhật các tham số cấu hình bằng cách gán thêm (hoặc ghi đè) 
        các cặp key-value mới vào không gian tên (namespace).
        
        Args:
            kwargs (dict): Dictionary chứa các cài đặt/biến cấu hình cần cập nhật.
        
        Returns:
            Đối tượng Config (self) hiện tại sau khi đã được cập nhật.
        """
        for name, value in kwargs.items():
            setattr(self.namespace, name, value)

        return self
