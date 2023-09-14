import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Any, Tuple
from itertools import product

from dotenv import load_dotenv
from selenium import webdriver
from selenium.common import ElementClickInterceptedException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from scrnshtr.util import trim_image, prepare_images, add_str_to_image

if os.environ.get('CHROMEDRIVER') is None:
    assert load_dotenv(Path('.', '.env'))
CHROMEDRIVER = os.environ.get('CHROMEDRIVER')

GECKODRIVER = os.environ.get('GECKODRIVER')


def timestamp():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())


@dataclass
class Screenshotter:
    url_server: str
    url_survey: str
    output_parent_path: Path
    url_port: Optional[str] = None
    token_list: List[str] = field(default_factory=list)
    viewport_widths_list: List[int] = field(default_factory=list)
    viewport_heights_list: List[int] = field(default_factory=list)
    pages_list: List[str] = field(default_factory=list)
    driver_name: str = 'chrome'
    driver: Union[webdriver.Chrome, webdriver.Firefox, None] = None
    options: Union[ChromeOptions, FirefoxOptions, None] = None
    flag_carousel: bool = True
    flag_click_on_next_and_screenshot: bool = False
    flag_end_when_done: bool = True
    flag_trim_image: bool = True
    flag_trim_in_place: bool = False
    flag_add_filename: bool = False
    flag_headless: bool = True
    flag_expand_accordion: bool = True
    language_suffix: str = 'de'
    url_protocol: str = 'http://'
    suffix_url: str = '.html'
    output_dir_name_format: str = '{ts}_screenshots_{survey}_{driver}'
    output_file_name_format: str = '{page}##{width}x{height}##{lang}.png'
    output_path: Optional[Path] = None
    languages_list: List[str] = field(default_factory=lambda: ['de'])
    dir_suffix: Optional[str] = None
    before_page: Optional[
        Callable[[Union[webdriver.Chrome, webdriver.Firefox], str, str, Optional[Dict[str, Any]]], None]] = None
    after_login: Optional[
        Callable[[Union[webdriver.Chrome, webdriver.Firefox], str, str, Optional[Dict[str, Any]]], None]] = None
    before_screenshot: Optional[
        Callable[[Union[webdriver.Chrome, webdriver.Firefox], str, Path, Optional[Dict[str, Any]], str], None]] = None
    after_login_args: Optional[Dict[str, Any]] = None
    before_page_args: Optional[Dict[str, Any]] = None
    before_screenshot_args: Optional[Dict[str, Any]] = None
    screenshot_files: Dict[str, List[Path]] = field(default_factory=dict)

    def base_url(self) -> str:
        base_url = self.url_protocol + self.url_server
        if self.url_port is not None:
            base_url += ':' + self.url_port
        base_url += '/' + self.url_survey
        return base_url

    def logout_url(self) -> str:
        return self.base_url() + '/j_spring_security_logout'

    def login_url(self, token: str) -> str:
        return self.base_url() + '/special/login.html?zofar_token=' + token

    def start_webdriver(self) -> None:
        if self.driver is None:
            if self.driver_name == 'chrome':
                assert CHROMEDRIVER is not None
                if self.options is None:
                    self.options = ChromeOptions()
                if self.flag_headless:
                    self.options.add_argument('--headless')

                # add first language as argument to chromedriver
                self.options.add_argument('--lang=' + self.languages_list[0])
                self.driver = webdriver.Chrome(CHROMEDRIVER, chrome_options=self.options)

            elif self.driver_name == 'firefox':
                assert GECKODRIVER is not None
                raise NotImplementedError('geckodriver not yet implemented')

            else:
                raise NotImplementedError(f'unknown driver type: {self.driver_name}')
        else:
            raise AssertionError('webdriver already present')

    def prepare_output_folder(self) -> Path:
        format_dict = {
            'ts': timestamp(),
            'driver': self.driver_name,
            'survey': self.url_survey
        }
        template = self.output_dir_name_format.format(**format_dict)
        if self.dir_suffix is None:
            out_path = Path(self.output_parent_path, template)
        else:
            out_path = Path(self.output_parent_path, template + '_' + self.dir_suffix)
        out_path.mkdir(parents=True, exist_ok=True)
        self.output_path = out_path
        return out_path.absolute()

    def set_viewport_size(self, width: int, height: int):
        self.driver.set_window_size(width, height)

    def run_screenshotter(self) -> Path:
        output_path = self.prepare_output_folder()
        self.start_webdriver()
        try:
            for token in self.token_list:
                self.driver.get(self.login_url(token))
                if self.after_login is not None:
                    self.after_login(self.driver, self.base_url(), self.suffix_url, self.after_login_args)
                for page in self.pages_list:
                    for lang in self.languages_list:
                        if self.before_page is not None:
                            self.before_page(self.driver, self.base_url(), self.suffix_url, self.before_page_args)
                        self.driver.get(f'{self.base_url()}/{page}{self.suffix_url}?zofar_lang={lang}')
                        for width, height in product(self.viewport_widths_list, self.viewport_heights_list):
                            self.set_viewport_size(width, height)
                            if self.before_screenshot:
                                self.before_screenshot(self.driver, page, self.output_path, self.before_screenshot_args,
                                                       lang)
                                if self.flag_expand_accordion:
                                    self.expand_accordion()
                            self.make_screenshot(page, width, height, lang)
                self.driver.get(self.logout_url())
        except Exception:
            traceback.print_exc()

        finally:
            self.driver.get(self.logout_url())
            self.driver.quit()
            return output_path

    def make_screenshot(self, page: str, width: int, height: int, lang: str):
        self.find_missing_labels(page, lang)
        self.find_visible_zofar_fn(page, lang)

        format_dict = {
            'n': str(self.pages_list.index(page)).zfill(len(str(len(self.pages_list)))),
            'page': page,
            'width': width,
            'height': height,
            'lang': lang
        }
        filename = self.output_file_name_format.format(**format_dict)
        out_file = Path(self.output_path, filename)

        self.driver.save_screenshot(str(out_file))

        self.register_screenshot(out_file, lang)

        if self.flag_trim_image:
            if self.flag_trim_in_place:
                trim_image(Path(out_file), Path(out_file))
            else:
                Path(self.output_path, 'cropped').mkdir(exist_ok=True)
                trim_image(Path(out_file), Path(self.output_path, 'cropped', filename))

        if self.flag_add_filename:
            add_str_to_image(out_file, os.path.splitext(filename)[0], "darkred", 'n', 16, 0)

    def register_screenshot(self, file: Path, lang: str):
        if lang not in self.screenshot_files:
            self.screenshot_files[lang] = []
        self.screenshot_files[lang].append(file)

    def find_missing_labels(self, page: str, lang: str):
        results_list = re.findall('[\?]{3}[A-Za-z0-9\._-äöüÄÖÜ]+[\?]{3}', self.driver.page_source)
        if results_list:
            results_str = '\n\t'.join(results_list)
            missing_string = f'# missing label on page "{page}", language "{lang}":\n\t{results_str}'
            self.save_logfile(missing_string, Path(self.output_path, 'errors.log'))

    def save_logfile(self, data: str, logfile: Path) -> None:
        log_data = ""
        if logfile.exists():
            log_data = logfile.read_text(encoding='utf-8') + '\n'
        log_data += data
        logfile.write_text(data=log_data, encoding='utf-8')

    def find_visible_zofar_fn(self, page: str, lang: str):
        html_root = self.driver.find_element(By.XPATH, '/html')
        assert isinstance(html_root, WebElement)

        results_list = re.findall('zofar\.', html_root.text)
        results_list += re.findall('layout\.', html_root.text)
        results_list += re.findall('\.value', html_root.text)
        results_list += re.findall('PRELOAD', html_root.text)

        if results_list:
            results_str = '\n\t'.join(results_list)
            missing_string = f'# zofar functions on page "{page}", language "{lang}":\n\t{results_str}'
            self.save_logfile(missing_string, Path(self.output_path, 'errors.log'))

    @staticmethod
    def side_by_side_images(zipped_files_tuples: List[Tuple[Path, Path]], output_parent_path: Path,
                            trim_result: bool) -> None:
        prepare_images(zipped_files_list=zipped_files_tuples,
                       output_path=Path(output_parent_path, timestamp() + '_compare'), trim_result=trim_result)

    def compare_languages(self, lang1: str, lang2: str, trim_result: bool = True):
        tuples_list = [tuple(e) for e in zip(self.screenshot_files[lang1], self.screenshot_files[lang2])]
        self.side_by_side_images(tuples_list, Path(self.output_path.parent, timestamp() + '_compare'),
                                 trim_result=trim_result)
        # prepare_images(zipped_files_list=tuples_list,
        #                output_path=Path(self.output_path.parent, timestamp() + '_compare'), trim_result=trim_result)

    def expand_accordion(self):
        accordion_elements = self.driver.find_elements(By.CSS_SELECTOR, '.containerAcc')
        if accordion_elements:
            runs = len(accordion_elements)
            for i in range(runs):
                all_elements = self.driver.find_elements(By.CSS_SELECTOR, '.containerAcc')
                if i > len(all_elements)-1:
                    break
                element = all_elements[i]
                countdown = 10
                while countdown > 0:
                    time.sleep(.3)
                    try:
                        element.click()
                        break
                    except ElementClickInterceptedException:
                        traceback.print_exc()
                    finally:
                        countdown -= 1
