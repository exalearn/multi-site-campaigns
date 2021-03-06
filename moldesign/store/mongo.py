"""Thin wrapper over MongoDB"""
from typing import List, Dict, Any, Set, Optional, Tuple

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection, UpdateResult
from flatten_dict import flatten

from moldesign.store.models import MoleculeData
from moldesign.store.recipes import apply_recipes


def generate_update(moldata: MoleculeData) -> Dict[str, Dict[str, Any]]:
    """Generate an update command for a certain molecule

    Returns:
        A dictionary ready for the "update_one" command
    """

    # Mark all of the fields that get "set"/overwritten
    set_fields = flatten(moldata.dict(exclude_defaults=True, exclude={'key', 'subsets'}), 'dot')
    output = {'$set': set_fields}

    # Mark all of the fields that get appended to
    extend_fields = flatten(moldata.dict(exclude_defaults=True, include={'subsets'}), 'dot')
    for key, value in extend_fields.items():
        if len(value) > 0:
            output['$addToSet'] = {key: {"$each": value}}

    return output


class MoleculePropertyDB:
    """Wrapper for a MongoDB holding molecular property data"""

    def __init__(self, collection: Collection):
        """
        Args:
            collection: Collection of molecule property documents
        """
        self.collection = collection

    @classmethod
    def from_connection_info(cls, hostname: str = "localhost", port: Optional[int] = None,
                             database: str = "edw", collection: str = "molecules", **kwargs):
        client = MongoClient(hostname, port=port, **kwargs)
        db = client.get_database(database)
        return MoleculePropertyDB(db.get_collection(collection))

    def initialize_index(self):
        """Prepare a new collection.

        Sets the InChI key field as a key index"""
        return self.collection.create_index('key', unique=True)

    def get_training_set(self, input_fields: List[str], output_fields: List[str]) \
            -> Dict[str, List[Any]]:
        """Gather a training set from the existing database.

        Get a set of entries where both the input and output exist.

        Args:
            input_fields: List of fields that must exist
            output_fields: Which fields to produce in the output fields
        Returns:
            Dictionary where keys of are the requested input and output fields
        """

        # Get the records
        query = dict((v, {'$exists': True}) for v in input_fields + output_fields)
        cursor = self.collection.find(query, input_fields + output_fields)

        # Compile the outputs
        # TODO (wardlt): Consider switching to Numpy arrays for these
        output = dict((v, []) for v in input_fields + output_fields)
        for record in cursor:
            record = flatten(record, 'dot')
            for i in input_fields + output_fields:
                output[i].append(record[i])
        return output

    def get_eligible_molecules(self, input_fields: List[str], output_fields: List[str]) \
            -> Dict[str, List[Any]]:
        """Gather molecules which are eligible for running a certain computation

        Determines eligibility if certain input_fields are populated and the
        output fields _are not set_.

        Args:
            input_fields: List of fields that must exist
            output_fields: Which fields must not exist
        Returns:
            A dictionary with keys of each input field. This will always include "identifier.inchi" and "key"
        """

        # Launch the query
        cursor, projection = self._query_eligible_molecules(input_fields, output_fields)

        # Gather the inputs
        inputs = dict((v, []) for v in projection)
        for record in cursor:
            record = flatten(record, 'dot')
            for i in projection:
                inputs[i].append(record[i])
        return inputs

    def get_eligible_molecule_records(self, input_fields: List[str], output_fields: List[str]) -> List[MoleculeData]:
        """Gather molecules which are eligible for running a certain computation

        Determines eligibility if certain input_fields are populated and the
        output fields _are not set_.

        Args:
            input_fields: List of fields that must exist
            output_fields: Which fields must not exist
        Returns:
            List of molecule data records. They only contain the fields listed
        """

        cursor, _ = self._query_eligible_molecules(input_fields, output_fields)
        return [MoleculeData.parse_obj(record) for record in cursor]

    def _query_eligible_molecules(self, input_fields, output_fields) -> Tuple[pymongo.cursor.Cursor, List[str]]:
        """Start a query for eligible molecules (see :meth:`get_eligible_molecules`)

        Args:
            input_fields: List of fields that must exist
            output_fields: Which fields must not exist
        Returns:
            - Cursor over matching records
            - Fields that should be present in each record
        """

        # Get the "exists" query fields
        projection = input_fields + ['identifier.inchi', 'key']
        projection = list(set(projection))  # Remove duplicates

        # Build the query
        query = dict((v, {'$exists': True}) for v in projection)
        for f in output_fields:
            query[f] = {'$exists': False}

        # Run the query
        cursor = self.collection.find(query, projection)
        return cursor, projection

    def update_molecule(self, molecule: MoleculeData) -> UpdateResult:
        """Update the data for a single molecule

        Args:
            molecule: Data for a certain molecule to be updated.
                All fields specified in this record will be updated or added to the matching document.
                No fields will be deleted by this operation.
        Returns:
            An update result
        """

        # Double-check the format
        MoleculeData.validate(molecule)

        # Update the derived properties
        molecule.update_thermochem()
        apply_recipes(molecule)

        # Generate the update and send it to the database
        update_record = generate_update(molecule)
        return self.collection.update_one({'key': molecule.key}, update_record, upsert=True)

    def get_molecules(self, match: Optional[Dict] = None, output_key: str = 'identifier.inchi') -> Set[str]:
        """Get all of the molecules in that match a certain query

        Returns a query of all of their object

        Args:
            match: Query used to filter down molecules
            output_key: Which field from the records to output
        Returns:
            Set of all of the unique values of that field
        """

        return set(self.collection.distinct(output_key, filter=match))

    def get_molecule_record(self, key: Optional[str] = None, smiles: Optional[str] = None, inchi: Optional[str] = None,
                            **kwargs) -> MoleculeData:
        """Get a record for a certain molecule

        Args:
            key: InChI key
            smiles: SMILES string (not, not a unique identifier!)
            inchi: InChI string (we store molecules with the stereochemistry block)
        Returns:
            All requested data for the molecule, if in database. Generates a fresh record if not available
        """

        # Make the query
        query = {}
        if key is not None:
            query['key'] = key
        for tag, value in [('smiles', smiles), ('inchi', inchi)]:
            if value is not None:
                query[f'identifier.{tag}'] = value

        # Return a document
        record = self.collection.find_one(query, **kwargs)
        if record is not None:
            return MoleculeData.parse_obj(record)

        # Make a fresh document
        if key is not None:
            return MoleculeData(key=key)
        return MoleculeData.from_identifier(**{'inchi': inchi, 'smiles': smiles})

    def update_derived_properties(self):
        """Update the derived properties for all molecules in a database"""

        cursor = self.collection.find({})
        for record in cursor:
            # Get a molecule
            data = MoleculeData.parse_obj(record)

            # Update its properties
            self.update_molecule(data)
