
class Space:
    def __init__(self):
        self.objects = []

    def add_object(self, obj):
        self.objects.append(obj)

    def get_objects(self, obj_type):
        return [obj for obj in self.objects if isinstance(obj, obj_type)]
    
    def __iter__(self):
        return iter(self.objects)  # Allow iteration over stored objects
