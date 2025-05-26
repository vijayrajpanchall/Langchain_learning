from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Vijay"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(default=0, gt=0, le=10)


new_student = {"age": "25", "email": "abc@gmail.com", "cgpa": 8.5}


student = Student(**new_student)
print(student)
