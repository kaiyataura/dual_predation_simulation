import cython
from collections import deque

maxR:cython.double = 0

@cython.cclass
class QuadPoint:
    x:cython.double
    y:cython.double
    r:cython.double
    data: object

    def __init__(self, x:cython.double, y:cython.double, r:cython.double, data: object):
        self.x = x
        self.y = y
        self.r = r
        self.data = data
        global maxR
        if r > maxR: maxR = r

    @cython.ccall
    def get(self):
        return self.data

@cython.cclass
class QuadTree:
    x:cython.double
    y:cython.double
    w:cython.double
    h:cython.double
    n:cython.int
    leaf:cython.bint
    data:list
    len:cython.int
    TL:QuadTree
    TR:QuadTree
    BL:QuadTree
    BR:QuadTree

    def __init__(self, x:cython.double, y:cython.double, w:cython.double, h:cython.double, n: cython.int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.n = n
        self.leaf = True
        self.data = [None] * n
        self.len = 0

    @cython.cfunc
    def contains(self, x:cython.double, y:cython.double) -> cython.bint:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h
    
    @cython.cfunc
    def intersects(self, x:cython.double, y:cython.double, w:cython.double, h:cython.double) -> cython.bint:
        return x < self.x + self.w and x + w > self.x and y < self.y + self.h and y + h > self.y

    @cython.cfunc
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def insertQPoint(self, qpoint:QuadPoint):
        if not self.leaf:
            if self.TL.contains(qpoint.x, qpoint.y): self.TL.insertQPoint(qpoint)
            elif self.TR.contains(qpoint.x, qpoint.y): self.TR.insertQPoint(qpoint)
            elif self.BL.contains(qpoint.x, qpoint.y): self.BL.insertQPoint(qpoint)
            elif self.BR.contains(qpoint.x, qpoint.y): self.BR.insertQPoint(qpoint)
        elif self.len >= self.n:
            self.TL = QuadTree(self.x,              self.y,              self.w / 2, self.h / 2, self.n)
            self.TR = QuadTree(self.x + self.w / 2, self.y,              self.w / 2, self.h / 2, self.n)
            self.BL = QuadTree(self.x,              self.y + self.h / 2, self.w / 2, self.h / 2, self.n)
            self.BR = QuadTree(self.x + self.w / 2, self.y + self.h / 2, self.w / 2, self.h / 2, self.n)
            self.leaf = False
            i: cython.int
            for i in range(self.len): self.insertQPoint(self.data[i])
            self.insertQPoint(qpoint)
        else: 
            self.data[self.len] = qpoint
            self.len += 1

    @cython.ccall
    def insert(self, x:cython.double, y:cython.double, r:cython.double, data:object) -> cython.void: 
        self.insertQPoint(QuadPoint(x, y, r, data))

    # def move(self, x:cython.double, y:cython.double, qpoint, parent) -> cython.void:
    #     if self.children:
    #         for child in self.children:
    #             if child.contains(qpoint.x, qpoint.y): child.move(x, y, qpoint, parent); return
    #     else: 
    #         qpoint.x = x
    #         qpoint.y = y
    #         if not self.contains(x, y):
    #             self.data.remove(qpoint)
    #             parent.insertQPoint(qpoint)
    
    @cython.cfunc
    def query(self, x1:cython.double, y1:cython.double, x2:cython.double, y2:cython.double, qpoint:cython.bint=False):
        result:deque = deque()
        x:cython.double = min(x1, x2)
        y:cython.double = min(y1, y2)
        if self.leaf:
            i: cython.int
            for i in range(self.len):   
                data: QuadPoint = self.data[i]
                if x <= data.x < max(x1, x2) and y <= data.y < max(y1, y2):
                    result.append(data if qpoint else data.data)
        else:
            w: cython.double = abs(x2 - x1)
            h: cython.double = abs(y2 - y1)
            if self.TL.intersects(x, y, w, h): 
                result.extend(self.TL.query(x1, y1, x2, y2, qpoint))
            if self.TR.intersects(x, y, w, h): 
                result.extend(self.TR.query(x1, y1, x2, y2, qpoint))
            if self.BL.intersects(x, y, w, h): 
                result.extend(self.BL.query(x1, y1, x2, y2, qpoint))
            if self.BR.intersects(x, y, w, h): 
                result.extend(self.BR.query(x1, y1, x2, y2, qpoint))
        return result

    @cython.ccall
    def query_all(self, qpoint:cython.bint=False):
        if self.leaf: return deque(self.data[0:self.len] if qpoint else [self.data[i].get() for i in range(self.len)])
        result:deque = deque()
        result.extend(self.TL.query_all(qpoint))
        result.extend(self.TR.query_all(qpoint))
        result.extend(self.BL.query_all(qpoint))
        result.extend(self.BR.query_all(qpoint))
        return result

    @cython.cfunc
    def query_radius(self, x:cython.double, y:cython.double, r:cython.double, qpoint:cython.bint=False):
        points:deque = self.query(x - r - maxR, y - r - maxR, x + r + maxR, y + r + maxR, True)
        result:deque = deque()
        point: QuadPoint
        for point in points:
            if (point.x - x) ** 2 + (point.y - y) ** 2 <= (point.r + r) ** 2:
                result.append(point if qpoint else point.data)
        return result

    @cython.ccall
    def query_pairs(self, qpoint:cython.bint=False):
        points:deque = self.query_all(True)
        result:deque = deque()
        point:QuadPoint
        other:QuadPoint
        for point in points:    
            for other in self.query_radius(point.x, point.y, point.r, True):
                if point != other: result.append((point, other) if qpoint else (point.data, other.data))
        return result