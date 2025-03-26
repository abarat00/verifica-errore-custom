from collections import deque
import numpy as np
import random
import torch

class Memory:
    """
    Uniform Experience Replay Memory.
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class Node:
    count = 0
    saturated = False

    def __init__(self, max_size, index_heap=None, l_child=None, r_child=None, children_heap=None, parent=None, parent_heap=None, value=0.0, sliding="oldest"):
        if children_heap is None:
            children_heap = []
        self.max_size = max_size
        self.index = Node.count
        self.index_heap = index_heap
        self.l_child = l_child
        self.r_child = r_child
        self.children_heap = sorted(children_heap, reverse=True)
        self.parent = parent
        self.parent_heap = parent_heap
        self.value = value
        self.leaf = (l_child is None) and (r_child is None)
        self.leaf_heap = len(children_heap) == 0
        self.complete = False
        if self.leaf:
            self.index = Node.count
            Node.count += 1
            self.level = 0
            Node.saturated = Node.count >= self.max_size
        elif self.r_child is None:
            self.level = self.l_child.level + 1
        else:
            self.level = min(self.l_child.level, self.r_child.level) + 1
            self.complete = (self.l_child.level == self.r_child.level)

    @staticmethod
    def reset_count():
        Node.count = 0
        Node.saturated = False

    def update_complete(self):
        assert not self.leaf, "Do not update the status of a leaf"
        if self.r_child is None:
            pass
        else:
            self.complete = (self.l_child.level == self.r_child.level)

    def update_level(self):
        if self.r_child is None:
            self.level = self.l_child.level + 1
        else:
            self.level = min(self.l_child.level, self.r_child.level) + 1

    def update_value(self):
        self.value = self.l_child.value + self.r_child.value

    def update(self):
        self.update_level()
        self.update_complete()
        self.update_value()

    def update_leaf_heap(self):
        self.leaf_heap = len(self.children_heap) == 0

    def set_l_child(self, l_child):
        self.l_child = l_child

    def set_r_child(self, r_child):
        self.r_child = r_child

    def set_children_heap(self, children_heap):
        self.children_heap = children_heap
        self.children_heap.sort(reverse=True)
        for child in children_heap:
            child.set_parent_heap(self)

    def replace_child_heap(self, child_origin, child_new):
        assert child_origin in self.children_heap, "The child you want to replace does not belong to the children of current node!"
        for i, child in enumerate(self.children_heap):
            if child == child_origin:
                self.children_heap[i] = child_new
        self.children_heap.sort(reverse=True)
        child_new.set_parent_heap(self)

    def add_child_heap(self, child):
        assert len(self.children_heap) < 2, "The node already has 2 children, cannot add a child; consider replacing."
        self.children_heap.append(child)
        self.children_heap.sort(reverse=True)
        child.set_parent_heap(self)

    def set_parent_heap(self, parent_heap):
        self.parent_heap = parent_heap

    def set_index_heap(self, index_heap):
        self.index_heap = index_heap

    def __lt__(self, node):
        return self.value < node.value


def retrieve_leaf(node, s):
    if node.leaf:
        return node.index
    elif node.l_child.value >= s:
        return retrieve_leaf(node.l_child, s)
    else:
        return retrieve_leaf(node.r_child, s - node.l_child.value)


retrieve_leaf_vec = np.vectorize(retrieve_leaf, excluded=set([0]))


def retrieve_value(node):
    return node.value


retrieve_value_vec = np.vectorize(retrieve_value)


class Heap:
    def __init__(self):
        self.track = []
        self.root = None
        self.last_child = None

    def swap(self, child, parent):
        child_children_heap, parent_children_heap, grand_parent = child.children_heap, parent.children_heap, parent.parent_heap
        child_index_heap, parent_index_heap = child.index_heap, parent.index_heap
        child.set_index_heap(parent_index_heap)
        parent.set_index_heap(child_index_heap)
        parent.set_children_heap(child_children_heap)
        child.set_children_heap(parent_children_heap)
        child.replace_child_heap(child, parent)
        if grand_parent is not None:
            grand_parent.replace_child_heap(parent, child)
        else:
            child.set_parent_heap(None)
            self.root = child
        self.track[child.index_heap] = child
        self.track[parent.index_heap] = parent

    def sift_up(self, node):
        parent = node.parent_heap
        changed = False
        while (parent is not None) and (node > parent):
            self.swap(node, parent)
            parent = node.parent_heap
            changed = True
        return changed

    def sift_down(self, node):
        children = node.children_heap
        changed = False
        while (len(children) != 0) and (children[0] > node):
            self.swap(children[0], node)
            children = node.children_heap
            changed = True
        return changed

    def update(self, node, value):
        value_prev = node.value
        node.value = value
        if value < value_prev:
            self.sift_down(node)
        else:
            self.sift_up(node)

    def insert(self, node):
        self.track.append(node)
        node.set_index_heap(len(self.track) - 1)
        if self.root is None:
            self.root = node
        else:
            parent = self.track[(node.index_heap - 1) // 2]
            parent.add_child_heap(node)


class SumTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self.sub_left = None
        self.parents = deque()
        self.children = deque()
        self.complete = False

    def add_leaf(self, node):
        if self.sub_left is None:
            self.sub_left = node
            self.complete = True
        else:
            root = Node(self.max_size, l_child=self.sub_left)
            self.sub_left.parent = root
            self.parents.appendleft(root)
            self.children.append(node)
            self.complete = False
            if len(self.parents) >= 2:
                self.parents[-1].l_child = self.children[-2]
                self.children[-2].parent = self.parents[-1]
                self.parents[-1].r_child = self.children[-1]
                self.children[-1].parent = self.parents[-1]
                self.parents[-1].update()
                while self.parents[-1].complete:
                    node = self.parents.pop()
                    self.children.pop()
                    self.children[-1] = node
                    if len(self.parents) == 1:
                        break
                    self.parents[-1].l_child = self.children[-2]
                    self.children[-2].parent = self.parents[-1]
                    self.parents[-1].r_child = self.children[-1]
                    self.children[-1].parent = self.parents[-1]
                    self.parents[-1].update()
                if len(self.parents) >= 2:
                    for i in range(-2, -len(self.parents), -1):
                        self.parents[i].l_child = self.children[i - 1]
                        self.children[i - 1].parent = self.parents[i]
                        self.parents[i].r_child = self.parents[i + 1]
                        self.parents[i + 1].parent = self.parents[i]
                        self.parents[i].update()
                    self.parents[0].r_child = self.parents[1]
                    self.parents[0].update()
                else:
                    self.parents[0].r_child = self.children[0]
                    self.children[0].parent = self.parents[0]
                    self.parents[0].update()
                    if self.parents[0].complete:
                        root = self.parents.pop()
                        self.children.pop()
                        self.sub_left = root
                        self.complete = True
            elif len(self.parents) == 1:
                self.parents[0].r_child = self.children[0]
                self.children[0].parent = self.parents[0]
                self.parents[0].update()
                if self.parents[0].complete:
                    root = self.parents.pop()
                    self.children.pop()
                    self.sub_left = root
                    self.complete = True

    def sample_batch(self, batch_size=64):
        root = self.sub_left if (len(self.parents) == 0) else self.parents[0]
        ss = np.random.uniform(0, root.value, batch_size)
        return retrieve_leaf_vec(root, ss)

    def update(self, node):
        parent = node.parent
        parent.update_value()
        parent = parent.parent
        while parent is not None:
            parent.update_value()
            parent = parent.parent

    def retrieve_root(self):
        return self.sub_left if len(self.parents) == 0 else self.parents[0]


def retrieve_first(couple):
    return couple[0]


retrieve_first_vec = np.vectorize(retrieve_first)


class PrioritizedMemory:
    """
    Prioritized Experience Replay Memory.
    """
    def __init__(self, max_size, sliding="oldest"):
        self.max_size = max_size
        assert sliding in ["oldest", "random"], "sliding parameter must be either 'oldest' or 'random'"
        self.sliding = sliding
        
        # Initialize buffer as two lists: experiences and corresponding nodes
        self.buffer = [[], []]
        for _ in range(max_size):
            self.buffer[0].append(None)
            self.buffer[1].append(None)
        
        self.tree = SumTree(max_size=max_size)
        self.heap = Heap()

    def update(self, index, priority):
        node = self.buffer[1][index]
        self.heap.update(node, priority)
        self.tree.update(node)

    def add(self, experience, priority):
        if not Node.saturated:
            leaf = Node(max_size=self.max_size, value=priority)
            index = leaf.index
            # Accediamo alle liste con la sintassi corretta:
            self.buffer[0][index] = experience
            self.buffer[1][index] = leaf
            self.tree.add_leaf(leaf)
            self.heap.insert(leaf)
        else:
            if self.sliding == "oldest":
                index = Node.count % self.max_size
                Node.count += 1
            elif self.sliding == "random":
                index = np.random.randint(0, self.max_size)
            leaf = self.buffer[1][index]
            self.buffer[0][index] = experience
            self.update(index, priority)

    def sample(self, batch_size):
        indices = self.tree.sample_batch(batch_size)
        experiences = [self.buffer[0][i] for i in indices]
        return experiences, indices
    
    def highest_priority(self):
        return self.heap.root.value

    def n_experiences(self):
        return len(self.heap.track)

    def sum_priorities(self):
        root = self.tree.retrieve_root()
        return root.value

    def retrieve_priorities(self, indices):
        leaves = [self.buffer[1][i] for i in indices]
        return retrieve_value_vec(np.array(leaves, dtype=object))