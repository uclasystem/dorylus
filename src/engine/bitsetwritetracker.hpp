// NOTE, copied from GraphChi

/**
 * @file
 * @author  Aapo Kyrola <akyrola@cs.cmu.edu>
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Aapo Kyrola, Guy Blelloch, Carlos Guestrin / Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 
 *
 * @section DESCRIPTION
 *
 * Bitset scheduler.
 */

#ifndef DEF_GRAPHCHI_BITSETWRITETRACKER
#define DEF_GRAPHCHI_BITSETWRITETRACKER

#include "densebitset.hpp"

class BitsetWriteTracker {
private:
	dense_bitset * previteration_bitset;
	dense_bitset * curiteration_bitset;
public:

	BitsetWriteTracker(IdType nvertices) {
		previteration_bitset = new dense_bitset(nvertices);
		curiteration_bitset = new dense_bitset(nvertices);

		previteration_bitset->clear();
		curiteration_bitset->clear();
	}

	void new_iteration() {
		dense_bitset * tmp = previteration_bitset;
		previteration_bitset = curiteration_bitset;
		curiteration_bitset = tmp;
		curiteration_bitset->clear();
	}

	virtual ~BitsetWriteTracker() {
		delete curiteration_bitset;
		delete previteration_bitset;
	}

	inline void write_task(IdType vertex) {
		curiteration_bitset->set_bit(vertex);
	}

	void resize(IdType maxsize) {
		previteration_bitset->resize(maxsize);
		curiteration_bitset->resize(maxsize);

	}

	inline bool was_written(IdType vertex) {
		return previteration_bitset->get(vertex);
	}

	void remove_tasks(IdType fromvertex, IdType tovertex) {
		curiteration_bitset->clear_bits(fromvertex, tovertex);
	}

	void write_task_to_all() {
		curiteration_bitset->setall();
		}

	IdType num_tasks() {
		IdType n = 0;
		for (IdType i = 0; i < previteration_bitset->size(); i++) {
			n += previteration_bitset->get(i);
		}
		return n;
	}

	dense_bitset* getPreviousBitset() const {
		return previteration_bitset;
	}

};

#endif

