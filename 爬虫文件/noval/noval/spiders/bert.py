import scrapy
from scrapy import signals
import re
import requests
from urllib.parse import urlencode
from noval.items import NovalItem, ChapterItem

class BertSpider(scrapy.Spider):
    name = 'bert'
    url = 'https://www.69zww.com/'
    num = 1
    tot_books = 0

    def id_to_url(self, id):
        return self.url + 'book_' + id + '/'

    def start_requests(self):
        self.start_urls = self.url + f'top/monthvisit/{self.num}.html'
        yield scrapy.Request(self.start_urls, callback=self.parse, dont_filter=True)

    def crawl_Chapter(self, response):
        reader_main = response.xpath('//*[@id="content"]')
        item = response.meta['item']
        item['content'] = reader_main.xpath('./text()').extract()[2:]
        yield item

    def crawl_book(self, response):
        book, url = response.meta['item']['book'], response.meta['item']['url']
        chapters = response.xpath('/html/body/div[4]/dl/dd/a')
        chap_urls = [url + chap for chap in chapters.xpath('./@href').extract()]
        chap_titles = chapters.xpath('./text()').extract()
        for id, url, chatitle in zip(range(100), chap_urls, chap_titles):
            item = ChapterItem(id=id, book=book, chapterTitle=chatitle)
            yield scrapy.Request(url, callback=self.crawl_Chapter, meta={'item': item})

    def parse(self, response):
        book_list = response.xpath('//*[@id="articlelist"]/ul[2]/li/span[@class="l2"]/a')
        self.tot_books += len(book_list)
        self.num += 1
        if self.tot_books < 500:
            yield scrapy.Request(self.url + f'top/monthvisit/{self.num}.html', callback=self.parse)
        books = book_list.xpath('./text()').extract()
        urls = book_list.xpath('./@href').extract()
        for i, book, url in zip(range(500), books, urls):
            print(book)
            item = NovalItem(book=book, url=url)
            yield item
            yield scrapy.Request(url, callback=self.crawl_book, meta={'item': item})