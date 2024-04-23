import Image from 'next/image'

import Datasets from './components/datasets'

export default function Home() {
  return (
    <main className="flex flex-col items-center ">
      <Datasets />
    </main>
  )
}
