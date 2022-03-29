
class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]
        def __len__(self):
            """ return number of elements """
            return self.max

        def pop(self):
            """ Pop the oldest element from the buffer and return it """
            if len(self.data) == 0:
                raise IndexError("pop from empty buffer")
            else:
                self.cur = (self.cur+1) % self.max
                return self.data.pop(0)


    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

    def __getitem__(self, ii):
        return self.data[ii]

    def __len__(self):
        """ Return the number of elements stored. """
        return len(self.data)

    def pop(self):
        """ Pop the oldest element from the buffer and return it """
        return self.data.pop(0)

