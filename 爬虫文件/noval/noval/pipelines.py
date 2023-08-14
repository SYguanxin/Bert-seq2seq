# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
from noval.items import NovalItem, ChapterItem


class NovalPipeline:
    def open_spider(self, spider):
        with open('data/new_train_data.json', 'r', encoding='utf-8')as fp:
            self.data = json.load(fp)

    def process_item(self, item, spider):
        if isinstance(item, NovalItem):
            self.data.update({f'{item["book"]}': []})
        if isinstance(item, ChapterItem):
            self.data[item['book']].append(
                {
                    'id': item['id'],
                    'title': item['chapterTitle'],
                    'content': item['content']
                }
            )
            print(item['chapterTitle'])

    def close_spider(self, spider):
        for k in self.data.keys():
            self.data[k] = sorted(self.data[k], key=lambda x: x['id'])
        with open('data/new_train_data.json', 'w', encoding='utf-8') as fp:
            json.dump(self.data, fp, ensure_ascii=False)