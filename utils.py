import random
class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        '''
        Randomly choose to
        1. Return input
        2. Return random image from pool and replace it with input 
        '''

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            if random.randint(0,1) == 1:
                return image
            else:
                idx = random.randint(0,len(self.images)-1)
                result = self.images.pop(idx)
                self.images.append(image)
                return result