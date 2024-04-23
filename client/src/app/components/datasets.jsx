'use client'

import React, { useState } from 'react'

export default function Datasets() {
  const [datasetInfo, setDatasetInfo] = useState('')

  const fetchDatasetInfo = async (dataset) => {
    const url = `http://localhost:5000/${dataset}`
    try {
      const response = await fetch(url)
      const data = await response.json()
      setDatasetInfo(data)
    } catch (error) {
      console.error('Error fetching dataset information:', error)
      setDatasetInfo({ error: 'Failed to load data.' })
    }
  }

  const renderConfusionMatrix = (confusionMatrix) => {
    return (
      <div className="mt-2">
        <div className="font-semibold text-center mb-2">Confusion Matrix:</div>
        <table className="mx-auto mb-4 text-gray-300">
          <thead>
            <tr>
              <th className=" px-4 py-2"></th> {/* Empty cell */}
              {confusionMatrix.map((_, index) => (
                <th key={index} className="px-4 py-2 text-gray-700">{index}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {confusionMatrix.map((item, index) => (
              <tr key={index}>
                <td className=" px-4 py-2 font-semibold text-gray-700">{index}</td>
                {item.row.map((value, valueIndex) => (
                  <td key={valueIndex} className="border border-gray-400 px-4 py-2">{value}</td>
                ))}
                <td className="px-4 py-2 text-left">{`→ ${item.class_name}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }


  const renderDatasetInfo = () => {
    if (!datasetInfo) {
      return <div className="text-sm text-gray-500">Select a dataset to view the information</div>
    }
    if (datasetInfo.error) {
      return <div className="text-sm text-red-500">{datasetInfo.error}</div>
    }
    return (
      <div className='p-6'>
        Number of examples: {datasetInfo.number_of_examples || 'N/A'}
        <br />
        Number of attributes: {datasetInfo.number_of_attributes || 'N/A'}
        <br />
        Number of classes: {datasetInfo.number_of_classes || 'N/A'}
        <br />
        Accuracy: {datasetInfo.accuracy || 'N/A'} ({datasetInfo.correctly_classified || 'N/A'} correctly classified)
        {datasetInfo.confusion_matrix ? renderConfusionMatrix(datasetInfo.confusion_matrix) : null}
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-black text-gray-300">
      <div>
        <button
          className="m-2 p-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded transition duration-300"
          onClick={() => fetchDatasetInfo('iris')}
        >
          Iris Dataset
        </button>
        <button
          className="m-2 p-2 bg-violet-500 hover:bg-violet-400 text-white rounded transition duration-300"
          onClick={() => fetchDatasetInfo('banknote')}
        >
          Banknote Dataset
        </button>
      </div>
      <div className="border border-gray-600 p-4 mt-6 w-full max-w-3xl bg-gray-800 rounded text-center text-sm font-mono leading-relaxed">
        {renderDatasetInfo()}
      </div>
    </div>
  )
}