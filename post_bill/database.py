#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import mysql.connector

logging.basicConfig(level=logging.INFO)


class DataBase(object):
    def __init__(self):
        self.user = 'doanota'
        self.password = 'Userlist@01'
        self.host = 'doanota.org'
        self.databse = 'doanota_sistema'
        self.logger = logging.getLogger(__name__)
        self.db_connection = None
        self.query = ("SELECT"
                      "           REPLACE(cp.qrcodecodigo, 'CFe', '') AS qrcodecodigo,"
                      "           cp.id,"
                      "           cp.cnpj,"
                      "           cp.coo,"
                      "           DATE_FORMAT(cp.data_cupom, '%d/%m/%Y') AS data_cupom,"
                      "           REPLACE(REPLACE(REPLACE(TRUNCATE(cp.valor, 2), '.', '#'),"
                      "                   ',',"
                      "                   '.'),"
                      "               '#',"
                      "               '') AS valor,"
                      "           et.razao_sefaz,"
                      "           cp.retract"
                      "       FROM"
                      "           cupons cp,"
                      "           entidades et"
                      "       WHERE"
                      "           1 = 1 AND et.estado = 'SP'"
                      "               AND cp.id_entidade = et.id_entidade"
                      "               and cp.statusprocessamento in (select status from status where subtipo='Aguardando SEFAZ')"
                      "               AND cp.retract <= 5"
                      "               AND et.razao_sefaz IS NOT NULL"
                      "       ORDER BY rand()"
                      "       LIMIT 1")

    def _connect(self):
        try:
            db_connection = mysql.connector.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                database=self.databse
            )
        except mysql.connector.errors.ProgrammingError:
            self.logger.error('Cannot connect to DB. Check you connection attributes.')
        else:
            self.logger.info('Connected to DB successfully')
            self.db_connection = db_connection

    def query_db(self):
        self._connect()
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute(self.query)
            rows = cursor.fetchall()
            if cursor.rowcount == 0:
                exit('There are no coupons in line.')
            query_update = ("update cupons set statusprocessamento=13,data_consumo=now(),retract=%s where id=%s")
            self.logger.info('Atualizando status 13 {0} {1}'.format(rows[0][7] + 1, rows[0][1]))
            data_update = (rows[0][7] + 1, rows[0][1])
            cursor.execute(query_update, data_update)
            self.db_connection.commit()
            cursor.close()

            return rows
        else:
            self.logger.error('Connection is not set. Check "db_connection"')
            exit('Execution is stopped.')

if __name__ == "__main__":
    d = DataBase()
    DATA = d.query_db()
