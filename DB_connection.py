import os
import cx_Oracle
import pandas as pd

class DB_Connection():

    def __init__(self):
        os.putenv('NLS_LANG', '.UTF8')
        # 연결에 필요한 기본 정보 (유저, 비밀번호, 데이터베이스 서버 주소)
        self.connection = cx_Oracle.connect("Ai_team1", "1234", "192.168.0.12:1521/orcl")
        self.cursor = self.connection.cursor()
        self.connection.autocommit = True

    def select_user(self):
        self.cursor.execute("select customer_id from CUSTOMERS where enter = 'True'")
        cnt = self.cursor.fetchone()
        print('cnt[0]',cnt[0])
        return cnt[0]

    def update_user(self, cnt):
        self.cursor.execute("update customers set enter = 'False' where  customer_id = :customer_id", {"customer_id":cnt})

    def insert_into_cart(self, customer_id, product_id):
        self.cursor.execute("insert into carts (cart_id, customer_id, product_id,cart_stock) values (cart_seq.nextval, :customer_id, :product_id, 1)",
                            {"customer_id":customer_id,'product_id':product_id})