#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from database import DataBase
from crack_captcha import crack_captcha

__author__ = 'Roman Byelyy'


class BasePage(object):
    url = None

    def __init__(self, driver):
        self.driver = driver
        self.driver.set_window_position(0, 0)
        self.driver.set_window_size(1024, 768)
        self.logger = logging.getLogger('selenium')

    def fill_field_by_css(self, css_selector, value):
        WebDriverWait(self.driver, 100).until(
            lambda driver: driver.find_element_by_css_selector(css_selector))
        element = self.driver.find_element_by_css_selector(css_selector)
        element.send_keys(value)

    def fill_field_by_id(self, id_selector, value):
        WebDriverWait(self.driver, 100).until(
            lambda driver: driver.find_element_by_id(id_selector))
        element = self.driver.find_element_by_id(id_selector)
        element.send_keys(value)

    def is_element_present(self, id_selector):
        WebDriverWait(self.driver, 100).until(
            lambda driver: driver.find_element_by_id(id_selector))
        try:
            self.driver.find_element_by_id(id_selector)  # find body tag element
        except NoSuchElementException:
            return False
        return True

    def navigate(self):
        self.driver.get(self.url)

    def close(self):
        self.driver.quit()


class Homepage(BasePage):
    url = "http://www.nfp.fazenda.sp.gov.br"

    def get_login_page(self):
        return Loginpage(self.driver)


class Loginpage(BasePage):
    url = "https://www.nfp.fazenda.sp.gov.br/login.aspx"

    def set_user_cnpj(self, user_cnpj):
        self.fill_field_by_id("UserName", user_cnpj)

    def set_password_cnpj(self, password_cnpj):
        self.fill_field_by_id("Password", password_cnpj)

    def submit(self):
        self.driver.find_element_by_id("Login").click()
        return Startpage(self.driver)


class Startpage(BasePage):
    url = "https://www.nfp.fazenda.sp.gov.br/Inicio.aspx"

    def get_registration_page(self):
        return Registratinopage(self.driver)


class Registratinopage(BasePage):
    captcha = True
    url = "https://www.nfp.fazenda.sp.gov.br/EntidadesFilantropicas/CadastroNotaEntidadeAviso.aspx"

    def select_combo_by_pattern(self, combo_value_pattern):
        self.driver.find_element_by_id("ctl00_ConteudoPagina_btnOk").click()
        element = self.driver.find_element_by_id("ddlEntidadeFilantropica")
        for option in element.find_elements_by_tag_name('option'):
            if combo_value_pattern in option.get_attribute("value"):
                option.click()
                break

    def confirm_registration(self):
        self.driver.find_element_by_id("ctl00_ConteudoPagina_btnNovaNota").click()
        try:
            if Registratinopage.captcha:
                self.driver.execute_script(
                    'document.querySelector("div.ui-dialog:nth-child(7) > div:nth-child(11) > div:nth-child(1) > button:nth-child(1)").click();')
                self.captcha = False
        except WebDriverException:
            self.logger.warn("Pop up dialog for {0} is not raised.".format(var_id))

    def download_captcha(self):
        if self.is_element_present('captchaNFP'):
            self.driver.get('https://www.nfp.fazenda.sp.gov.br/imagemDinamica.dContent?131157381060756280')
            self.driver.save_screenshot(self.driver.session_id + ".png")
            self.driver.execute_script("window.history.go(-1)")
            if self.is_element_present('errorTryAgain'):
                self.driver.find_element_by_id('errorTryAgain').send_keys("\n")
                alert = self.driver.switch_to_alert()
                alert.accept()
            self.logger.info("CAPTCHA is downloaded successfully")
        else:
            self.logger.warn("CAPTCHA was not found")

    def fill_form(self):
        if var_qrcodecodigo is None:
            var_cnpj_sub = re.sub('[-./]', '', var_cnpj)
            self.fill_field_by_css('#divCNPJEstabelecimento>input', var_cnpj_sub)
            self.fill_field_by_css('#divtxtDtNota>input', var_data_cupom.replace('/', ''))
            self.fill_field_by_css('#divtxtNrNota>input', var_coo)
            self.fill_field_by_css('#divtxtVlNota>input', var_valor)
        else:
            try:
                self.driver.find_element_by_class_name("text").click()
            except NoSuchElementException:
                self.logger.error('Unable to locate element: {"method":"class name","selector":"text"}')
            self.driver.find_element_by_class_name("text").send_keys(var_qrcodecodigo)

        if self.is_element_present('captchaNFP'):
            try:
                self.fill_field_by_id('ImagemRand', var_captcha)
            except UnicodeDecodeError:
                self.logger.error('CAPTCHA was not was not found on the registration page')

    def submit(self):
        self.driver.find_element_by_id("btnSalvarNota").send_keys("\n")

    def get_error(self):
        if self.is_element_present('lblErro'):
            self.logger.error("var_id({0}) FAILED DUE TO ERROR: '{1}'".format(var_id, self.driver.find_element_by_id(
                'lblErro').text.encode('utf-8').strip()))
        elif self.is_element_present('.ui-widget-header-error'):
            print (self.driver.find_element_by_css_selector('.ui-widget-header-error').text)
            return True
        else:
            return False


if __name__ == "__main__":
    logger = logging.getLogger('main')

    # Connect to DB
    db = DataBase()
    DATA = db.query_db()
    var_qrcodecodigo = DATA[0][0]
    var_id = DATA[0][1]
    var_cnpj = DATA[0][2]
    var_coo = DATA[0][3]
    var_data_cupom = DATA[0][4]
    var_valor = DATA[0][5]
    var_razao_sefaz = DATA[0][6]
    var_retract = DATA[0][7]
    logger.info('Current var_id is {0}'.format(var_id))
    if var_qrcodecodigo is not None:
        logger.info('.........QRCODIGO')
    else:
        logger.info('.........SEEMS NORMAL')

    # Set browser
    ff = webdriver.Firefox()

    # Login
    homepage = Homepage(driver=ff)
    homepage.navigate()
    loginpage = homepage.get_login_page()
    loginpage.navigate()
    loginpage.set_user_cnpj('72854642015')
    loginpage.set_password_cnpj('userlist')
    startpage = loginpage.submit()

    # Registration
    registrationpage = startpage.get_registration_page()
    registrationpage.navigate()
    registrationpage.select_combo_by_pattern(var_razao_sefaz)
    registrationpage.confirm_registration()
    registrationpage.download_captcha()

    # Crack CAPTCHA
    cc = crack_captcha.CrackCaptcha(input_image=registrationpage.driver.session_id + '.png',
                                    output_image='output_' + registrationpage.driver.session_id + '.png')
    cc.applay_otsu_gaussian()
    cc.save_cleaned_image()
    var_captcha = cc.execute_tesseract()

    # Registration
    registrationpage.fill_form()
    registrationpage.submit()

    if registrationpage.get_error():
        logger.error('Bill {0} is not created. Failed.'.format(var_id))
    else:
        logger.info('Bill {0} is created. Success.'.format(var_id))

    #registrationpage.close()
