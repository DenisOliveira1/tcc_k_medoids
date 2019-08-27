import random
import sys

class KMedoids:
    def __init__(self, n_cluster=2):
        self.n_cluster = n_cluster
        self.medoids = []
        self.clusters = {}
        self.medoid_update = True
        self.saver = {}
        self.saver_use_count = 0
        self.saver_use_skip = 0

##################### methods #####################

    def get_medoids(self):
        return sorted(self.medoids)

    def get_clusters(self):
        clusters = []
        for key,value in self.clusters.items():
            clusters.append(value.copy())
        for m in range(len(self.medoids)):
            clusters[m].append(self.medoids[m])
        return clusters

##################### fit #####################

    def _initial_medoids(self, instances):
        length = len(instances)-1
        medoids = []
        for i in range(self.n_cluster):
            n = random.randint(0,length)
            while(n in medoids):
                n = random.randint(0,length)
            medoids.append(n)
            self.clusters[n] = []
        return medoids

    def fit(self, instances):
        self.medoids = self._initial_medoids(instances)
        while(self.medoid_update):
            self._assigning(instances)
            self._optimizing(instances)
    
    #step 1
    def _assigning(self, instances):
        for i in range(len(instances)):
            if i not in self.medoids:
                lowest_dis = sys.maxsize
                for medoid in self.medoids:
                    current_dis = self._dis_cos(instances[medoid],instances[i])
                    if current_dis < lowest_dis:
                        lowest_dis = current_dis
                        lowest_dis_m = medoid
                self._remove_instance_from_all_clusters(i)
                self.clusters[lowest_dis_m].append(i)
        return None

    #step 2    
    def _optimizing(self, instances):
        self.medoid_update = False
        for key,value in self.clusters.items():
            current_m = key
            for i in value:
                current_m_dis = 0
                for z in value:
                    current_m_dis = current_m_dis + self._dis_cos(instances[current_m], instances[z])
                candidate_m = i
                candidate_dis = 0
                for y in value:
                    if y != i:
                        candidate_dis = candidate_dis + self._dis_cos(instances[candidate_m], instances[y])
                candidate_dis = candidate_dis + self._dis_cos(instances[candidate_m], instances[current_m])
                if candidate_dis < current_m_dis:
                    self._update_medoid(current_m, candidate_m)
                    current_m = candidate_m
        return None
    
##################### fit aux #####################

    #avoid ghost instances on step 1 and _update_medoid
    def _remove_instance_from_all_clusters(self, i):
        for key,value in self.clusters.items():
            remove_index = []
            for y in range(len(value)):
                if value[y] == i:
                    remove_index.append(y)
            for y in remove_index:                 
                value.pop(y)
        return None

    #update medoid if there is any update on step 2
    def _update_medoid(self, old, new):
        self._remove_instance_from_all_clusters(new) # remove new medoid from all clusters
        for m in range(len(self.medoids)):
            if self.medoids[m] == old:
                self.medoids[m] = new # update medoid in self.medoids
        for key,value in self.clusters.items():
            if key == old: 
                value.append(old) # add old medoid in the cluster it used to represent
        self.clusters[new] = self.clusters.pop(old)# update medoid name in self.clusters
        self.medoid_update = True

    def _save_dis(self, x1, x2, result):
        name = str(x1)+","+str(x2)
        self.saver[name] = result
        return None
    
    def _get_saved_dis(self, x1, x2):
        name = str(x1)+","+str(x2)
        return self.saver[name]

##################### dissimilarity #####################
   
    def _dis_cos(self, x1, x2):
        try:
            result = self._get_saved_dis(x1, x2)
            self.saver_use_count = self.saver_use_count + 1
        except:
            try:
                result = self._get_saved_dis(x2, x1)
                self.saver_use_count = self.saver_use_count + 1
            except:
                result = self._dis_cos_product(x1,x2)/(self._dis_cos_len(x1)*self._dis_cos_len(x2))
                result = 1 - result
                self._save_dis(x1, x2, result)
                self.saver_use_skip = self.saver_use_skip + 1
        return result

    def _dis_cos_product(self, x1, x2):
        result = 0
        for i in range(len(x1)):
            result = result + x1[i] * x2[i]
        return result  

    def _dis_cos_len(self, x):
        result = 0
        for attr in x:
            result = result + attr ** 2
        return result ** (1/2)

    def _dis_euclidean(self, x1, x2):
        result = 0
        for i in range(len(x1)):
            result = result + (x1[i] + x2[i]) ** 2
        return result ** (1/2)

##################### main #####################

if __name__ == "__main__":
    a = KMedoids(2)
    a.fit([[2, 6], [3, 4], [3, 8], [4, 7], [6, 2], [6, 4], [7, 3], [7, 4], [8, 5], [7, 6]])
    print (a.get_medoids())
    print (a.get_clusters())
