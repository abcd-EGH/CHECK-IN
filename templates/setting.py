import sqlite3

# 데이터베이스 파일 연결 (파일이 없으면 새로 생성됩니다)
conn = sqlite3.connect('image_db_sample.db')

# 커서 객체 생성
cursor = conn.cursor()

# 데이터베이스 사용
# 예: 테이블 생성
cursor.execute('''CREATE TABLE IF NOT EXISTS example_table
               (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

# 데이터 삽입
cursor.execute("INSERT INTO example_table (name, age) VALUES ('Alice', 21)")

# 변경사항 커밋
conn.commit()

# 데이터 조회
cursor.execute("SELECT * FROM example_table")
rows = print(cursor.fetchall())
for row in fows:
    print(row)
    
# 연결 종료
conn.close()