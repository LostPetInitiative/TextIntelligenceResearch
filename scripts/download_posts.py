import vk_api

vk_session = vk_api.VkApi()
vk = vk_session.get_api()

response  = vk.wall.get(count=1, owner_id=-37814238,access_token='')

if response['items']:
        print(response['items'][0])