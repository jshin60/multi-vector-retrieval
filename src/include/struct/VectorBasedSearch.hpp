//
// Created by 13172 on 2023/12/20.
//

#ifndef VECTORSETSEARCH_VECTORBASEDSEARCH_HPP
#define VECTORSETSEARCH_VECTORBASEDSEARCH_HPP
namespace VectorSetSearch {

    class CentroidDistance {
    public:
        uint32_t centroidID;
        uint32_t query_vecsID;
        float distance;

        CentroidDistance(uint32_t centroidID, uint32_t query_vecsID, float distance) {
            this->centroidID = centroidID;
            this->query_vecsID = query_vecsID;
            this->distance = distance;
        }
    };

}
#endif //VECTORSETSEARCH_VECTORBASEDSEARCH_HPP
