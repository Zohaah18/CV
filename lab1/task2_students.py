students = {
    "S001": {
        "Name": "Ali",
        "Major": "Computer Science",
        "Grades": [85, 90, 88, 92]
    },
    "S002": {
        "Name": "Sara",
        "Major": "Software Engineering",
        "Grades": [90, 95, 91, 89]
    },
    "S003": {
        "Name": "Ahmed",
        "Major": "Computer Science",
        "Grades": [78, 82, 80, 85]
    },
    "S004": {
        "Name": "Ayesha",
        "Major": "Data Science",
        "Grades": [94, 91, 96, 93]
    }
}

def highest_average_student(students):
    highest_average = -1
    highest_student = None
    for student_id, record in students.items():
        grades = record["Grades"]
        if len(grades) == 0:
            continue
        average = sum(grades) / len(grades)
        if average > highest_average:
            highest_average = average
            highest_student = record["Name"]

    return highest_student, highest_average

def search_by_major(students, major):
    found = False
    print(f"\nStudents in {major}:")
    for student_id, record in students.items():
        if record["Major"].lower() == major.lower():
            print(
                f"ID: {student_id}, "
                f"Name: {record['Name']}"
            )
            found = True

    if not found:
        print("No students found.")

student, average = highest_average_student(students)

print("Student with highest average:", student)
print("Highest average:", average)

search_by_major(students, "Computer Science")