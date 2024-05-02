from collections import deque

class QuadPoint:
    def __init__(self, x, y, r, data):
        self.x = x
        self.y = y
        self.r = r
        self.data = data

    def __str__(self):
        return f'<{self.x}, {self.y}, {self.data}>'

class QuadTree:
    maxR = 0
    def __init__(self, x, y, w, h, n):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.n = n
        self.data = deque()
        self.children = None

    def contains(self, x, y):
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h
    
    def intersects(self, x, y, w, h):
        return x < self.x + self.w and x + w > self.x and y < self.y + self.h and y + h > self.y

    def insertQPoint(self, qpoint):
        if self.children:
            for child in self.children:
                if child.contains(qpoint.x, qpoint.y): child.insertQPoint(qpoint); return
        elif len(self.data) >= self.n:
            self.children = (QuadTree(self.x,              self.y,              self.w / 2, self.h / 2, self.n),
                             QuadTree(self.x + self.w / 2, self.y,              self.w / 2, self.h / 2, self.n),
                             QuadTree(self.x,              self.y + self.h / 2, self.w / 2, self.h / 2, self.n),
                             QuadTree(self.x + self.w / 2, self.y + self.h / 2, self.w / 2, self.h / 2, self.n))
            temp = self.data
            self.data = None
            for d in temp: self.insertQPoint(d)
            self.insertQPoint(qpoint)
        else: self.data.append(qpoint); self.maxR = max(self.maxR, qpoint.r)

    def insert(self, x, y, r, data): 
        self.insertQPoint(QuadPoint(x, y, r, data))

    def move(self, x, y, qpoint, parent):
        if self.children:
            for child in self.children:
                if child.contains(qpoint.x, qpoint.y): child.move(x, y, qpoint, parent); return
        else: 
            qpoint.x = x
            qpoint.y = y
            if not self.contains(x, y):
                self.data.remove(qpoint)
                parent.insertQPoint(qpoint)

    def query(self, x1, y1, x2, y2, qpoint=False):
        result = deque()
        if self.children:
            for child in self.children:
                if child.intersects(min(x1, x2), min(y1, y2), abs(x2 - x1), abs (y2 - y1)):
                    result.extend(child.query(x1, y1, x2, y2, qpoint))
        else:
            for data in self.data:
                if min(x1, x2) <= data.x < max(x1, x2) and min(y1, y2) <= data.y < max(y1, y2):
                    result.append(data if qpoint else data.data)
        return result

    def query_all(self, qpoint=False):
        if not self.children: 
            return deque(self.data if qpoint else map(lambda p: p.data, self.data))
        result = deque()
        for child in self.children:
            result.extend(child.query_all(qpoint))
        return result

    def query_radius(self, x, y, r, qpoint=False):
        points = self.query(x - r - self.maxR, y - r - self.maxR, x + r + self.maxR, y + r + self.maxR, True)
        result = deque()
        for point in points:
            if (point.x - x) ** 2 + (point.y - y) ** 2 <= (point.r + r) ** 2:
                result.append(point if qpoint else point.data)
        return result

    def query_pairs(self, qpoint=False):
        points = self.query_all(True)
        
        result = deque()
        for point in points:    
            for other in self.query_radius(point.x, point.y, point.r, True):
                if point != other: result.append((point, other) if qpoint else (point.data, other.data))
        return result

    def __str__(self):
        if self.children: return str(list(map(str, self.children)))
        return str(self.data)
