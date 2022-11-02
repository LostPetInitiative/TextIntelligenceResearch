import vk_api
from decouple import config
import json
from datetime import datetime


def auth_handler():
    key = input("Enter authentication code: ")
    remember_device = True
    return key, remember_device


def wall_get(domain):
    """
    This function uses vk api to download information about posts oт the wall
    of the chosen group

    :param domain: the name of the vk group
    :return: writes a json file to disk (post_id, time, text, list of images as urls)
    """
    
    vk_session = vk_api.VkApi(token=config('TOKEN'))

    try:
        vk = vk_session.get_api()
    except vk_api.AuthError as error_msg:
        print(error_msg)
        return

    start = datetime.now()
    wall_contents = {}

    """
    wall.get — 5000 calls per day allowed
    offset — used to get posts that are below top 100
    """
    c = 0
    v = 0
    v_prev = None
    ads = []
    photo_ids = set()
    video_ids = set()
    post_ids = set()

    for i in range(0, 5, 1):
        post = vk.wall.get(domain=domain, count=100, filter=all, offset=i * 100)['items']
        try:
            for item in post:
                duplicate = False
                c += 1
                if item['id'] in post_ids:
                    print("Duplicate post\n")
                    continue
                post_ids.add(item['id'])
                if item['marked_as_ads'] == 1:
                    ads.append((item['id']))
                if 'attachments' in item:
                    v += 1
                    photos = set()
                    post_id = item['id']
                    date = item['date']
                    text_content = item['text']

                    for images in item['attachments']:
                        if images['type'] == 'photo':
                            # check if duplicate, but it is unreliable, because if post is reuploaded it gets new id 
                            # so catches only reposts
                            if images['photo']['id'] in photo_ids:
                                print(f"Duplicate photo {item['id']}")
                                duplicate = True
                                continue
                            photo_ids.add(images['photo']['id'])
                            try:
                                photos.add(images['photo']['sizes'][-1]['url'])
                            except IndexError:
                                pass

                        # working with video, gets a thumbnail of video file as image
                        elif images['type'] == 'video':
                            if images['video']['id'] in video_ids:
                                print(f"Duplicate video {item['id']}")
                                duplicate = True
                                continue
                            video_ids.add(images['video']['id'])
                            resolutions = []
                            for key, value in images['video'].items():
                                if key.startswith('photo_'):
                                    resolutions.append(key.split('_')[1])
                            res = f'photo_{max(resolutions)}'
                            photos.add(images['video'][res])

                    # checks if there were any photos in the post, because we are not interested in post with text only
                    # also stops from adding exact duplicates
                    if photos != set() and not duplicate:
                        wall_contents[post_id] = {
                            'date': date,
                            'text': text_content,
                            'images': list(photos)
                        }
            # check if no new posts are added during loop, meaning that the whole group is traversed
            if v_prev == v:
                break
            print(f'{v} posts completed')
            v_prev = v

        except KeyError:
            print(f'KeyError {item}')
            continue

        except Exception as e:
            with open('error.txt', 'w') as f:
                f.write(f'{c}, {item}')
            print(f'{e} occurred on post {c}\n')
            return

    with open(f'wall_{domain}.json', 'w', encoding='utf8') as json_file:
        json.dump(wall_contents, json_file, ensure_ascii=False)

    finish = datetime.now()
    print(f"Time elapsed: {(finish - start).total_seconds()}")
    print(ads)


if __name__ == '__main__':
    group_names = [
        'gefundendog',
        'verlorenkatze',
        'naidimoskva',
        'we_will_find',
        'poterjashkansk',
        'findcatsha',
        'public191189454',
        'poisk_v_nn',
        'poisksochi',
        'poiski.krasnodar',
        'poiski.rostov',
        'club96670880',
        'propalavtveri',
        'poiskzooru'
    ]
    for domain in group_names:
        wall_get(domain)

